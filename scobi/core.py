"""scobi core"""
import numpy as np
import os
from gymnasium import spaces, Env
import scobi.environments.env_manager as em
from scobi.utils.game_object import get_wrapper_class
from scobi.focus import Focus
from scobi.utils.logging import Logger
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import Box2D
from scobi.utils.game_object import LunarLanderObject


class Environment(Env):
    def __init__(self, env_name, seed=None, focus_dir="resources/focusfiles", focus_file=None, reward=0, hide_properties=False, silent=False, refresh_yaml=True, draw_features=False):
        print(f"[DEBUG] Initializing Environment with env_name: {env_name}, seed: {seed}")
        if env_name == "ALE/LunarLander-v5":
            os.environ["SCOBI_OBJ_EXTRACTOR"] = "LunarLander"
        else:
            os.environ["SCOBI_OBJ_EXTRACTOR"] = "OC_ATARI"
        self.logger = Logger(silent=silent)
        self.oc_env = em.make(env_name, self.logger)
        print(f"[DEBUG] Base environment created for {env_name}")
        self.seed = seed
        self.randomstate = np.random.RandomState(self.seed)
        # TODO: tie to em.make
        self.game_object_wrapper = get_wrapper_class()
        print(f"Self.game_object_wrapper:   {self.game_object_wrapper}")

        # TODO: oc envs should answer this, not the raw env
        if env_name == "ALE/LunarLander-v5":
            actions = [str(i) for i in range(self.oc_env.unwrapped.action_space.n)]
        else:
            actions = self.oc_env._env.unwrapped.get_action_meanings()

        self.oc_env.reset(seed=self.seed)
        self.noisy_objects = os.environ["SCOBI_OBJ_EXTRACTOR"] == "Noisy_OC_Atari"

        if env_name == "ALE/LunarLander-v5":
            max_objects = self.extract_all_objects()
        elif hasattr(self.oc_env, 'max_objects'):
            max_objects = self._wrap_map_order_game_objects(self.oc_env.max_objects, env_name, reward)
        else:
            max_objects = []

        self.did_reset = False
        self.focus = Focus(env_name, reward, hide_properties, focus_dir, focus_file, max_objects, actions, refresh_yaml, self.logger)
        self.focus_file = self.focus.FOCUSFILEPATH
        self.action_space = spaces.Discrete(len(self.focus.PARSED_ACTIONS))
        self.action_space_description = self.focus.PARSED_ACTIONS
        self.observation_space_description = self.focus.PARSED_PROPERTIES + self.focus.PARSED_FUNCTIONS #this and feature_vector_desc is redundant
        self.feature_vector_description = self.focus.get_feature_vector_description()
        self.num_envs = 1
        self.draw_features = draw_features
        self.feature_attribution = []
        self.render_font = ImageFont.truetype(str(Path(__file__).parent / 'resources' / 'Gidole-Regular.ttf'), size=38)
        self._obj_obs = None  # observation augmented with objects
        self._rel_obs = None  # observation augmented with relations
        self._top_features = []

        self.original_obs = []
        self.original_reward = []
        self.ep_env_reward = None
        self.ep_env_reward_buffer = 0
        self.reset_ep_reward = True

        if reward == 2: # mix rewards
            self._reward_composition_func = lambda a, b : a + b
        elif reward == 1: # scobi only
            self._reward_composition_func = lambda a, b : a
        else: # env only
            self._reward_composition_func = lambda a, b : b

        if self.noisy_objects:
            self.logger.GeneralInfo("Using noisy object detection (default: std 3, detection error rate 5%)")

        self.reset()
        self.step(0) # step once to set the feature vector size
        self.observation_space = spaces.Box(low=-2**63, high=2**63 - 2, shape=(self.focus.OBSERVATION_SIZE,), dtype=np.float32)
        if env_name == "ALE/LunarLander-v5":
            self.ale = None
        else:    
            self.ale = self.oc_env._env.unwrapped.ale
        self.reset()
        self.did_reset = False # still require user to properly call a (likely seeded) reset()

    def extract_all_objects(self):
        """Extrahiere alle potenziellen Objekte aus der Umgebung."""
        objects = []
        for attr_name in dir(self.oc_env.unwrapped):
            if "drawlist" in attr_name:  # Überspringe drawlist-Attribute
                continue
            attr = getattr(self.oc_env.unwrapped, attr_name, None)
            # Filtere physische Objekte (b2Body) und extrahiere Position
            if isinstance(attr, Box2D.b2Body):
                position = getattr(attr, 'position', (0, 0))
                objects.append(LunarLanderObject(name=attr_name, position=(position.x, position.y)))
            elif isinstance(attr, list):  # Beispiel: Beine oder Partikel
                for i, sub_attr in enumerate(attr, start=1):
                    if isinstance(sub_attr, Box2D.b2Body):
                        position = getattr(sub_attr, 'position', (0, 0))
                        objects.append(LunarLanderObject(name=f"{attr_name}_{i}", position=(position.x, position.y)))
        return objects    

    def step(self, action):
        if not self.did_reset:
            self.logger.GeneralError("Cannot call env.step() before calling env.reset()")
        elif self.action_space.contains(action):
            obs, reward, truncated, terminated, info = self.oc_env.step(action)
            if self.focus.ENV_NAME == "LunarLander-v5":
                objects = self.extract_all_objects()
            else:    
                objects = self._wrap_map_order_game_objects(self.oc_env.objects, self.focus.ENV_NAME, self.focus.REWARD_SHAPING)
            sco_obs, sco_reward = self.focus.get_feature_vector(objects)
            freeze_mask = self.focus.get_current_freeze_mask()
            if self.draw_features:
                self._obj_obs = self._draw_objects_overlay(obs)
                self._rel_obs = self._draw_relation_overlay(obs, sco_obs, freeze_mask, action)
            self.original_obs = obs
            self.original_reward = reward
            self.ep_env_reward_buffer += self.original_reward
            if self.reset_ep_reward:
                self.ep_env_reward = None
                self.reset_ep_reward = False
            if terminated or truncated:
                self.ep_env_reward = self.ep_env_reward_buffer
                self.ep_env_reward_buffer = 0
                self.reset_ep_reward = True
                self.focus.reward_subgoals = 0
            final_reward = self._reward_composition_func(sco_reward, reward)
            # self.sco_obs = sco_obs
            return sco_obs, final_reward, truncated, terminated, info # 5
        else:
            raise ValueError("scobi> Action not in action space")

    def reset(self, *args, **kwargs):
        self.did_reset = True
        # additional scobi reset steps here
        self.focus.reward_threshold = -1
        self.focus.reward_history = [0, 0]
        _, info = self.oc_env.reset(*args, **kwargs)
        if self.focus.ENV_NAME == "LunarLander-v5":
            objects = self.extract_all_objects()
        else:
            objects = self._wrap_map_order_game_objects(self.oc_env.objects, self.focus.ENV_NAME, self.focus.REWARD_SHAPING)
        sco_obs, _ = self.focus.get_feature_vector(objects)
        # self.sco_obs = sco_obs
        return sco_obs, info

    def close(self):
        # additional scobi close steps here
        self.oc_env.close()

    def set_feature_attribution(self, att):
        self.feature_attribution = att

    def _wrap_map_order_game_objects(self, oc_obj_list, env_name, reward_shaping):
        out = []
        counter_dict = {}
        player_obj = None

        # wrap
        if self.noisy_objects:
            scobi_obj_list = [self.game_object_wrapper(obj, std=3, error_rate=0.05, random_state=self.randomstate) for obj in oc_obj_list]
        else:
            scobi_obj_list = [self.game_object_wrapper(obj) for obj in oc_obj_list]
        #if self.noisy_objects:
        #    for o in scobi_obj_list:
        #        o.add_noise(std=3, error_rate=0.05, random_state=self.randomstate)


        # order
        for scobi_obj in scobi_obj_list:
            if "Player" in scobi_obj.name or "Chicken" in scobi_obj.name:
                player_obj = scobi_obj
                break

        if "Kangaroo" in env_name and reward_shaping != 0:
            scales = []
            rest = []
            for x in scobi_obj_list:
                scales.append(x) if "Scale" in x.name else rest.append(x)
            scales = sorted(scales, key=lambda a : abs(a.y_distance(player_obj)))
            rest = sorted(rest, key=lambda a : a.distance(player_obj))
            scobi_obj_list = rest + scales
        else:
            scobi_obj_list = sorted(scobi_obj_list, key=lambda a : a.distance(player_obj))

        # map
        for scobi_obj in scobi_obj_list:
            if not scobi_obj.category in counter_dict:
                counter_dict[scobi_obj.category] = 1
            else:
                counter_dict[scobi_obj.category] +=1
            scobi_obj.number = counter_dict[scobi_obj.category]
            out.append(scobi_obj)

        # returns full objectlist [closest_visible : farest_visible] + [closest_invisible : farest invisibe]
        # if focus file specifies for example top3 closest objects (by selecting pin1, pin2, pin3),
        # the features vector is always calculated based on the 3 closest visible objects of category pin.
        # so if for example pin1 becomes invisible during training, the top3 list is filled accordingly with closest visible objects of category pin
        # if there is none to fill, pin2 and pin3 are the closest visible and all derived features for the third pin (which is invisble) are frozen
        return out


    # def _mark_bb(self, image_array, bb, color=(255, 0, 0), surround=True):
    #     """
    #     marks a bounding box on the image
    #     """
    #     x, y, w, h = bb
    #     x = int(x)
    #     y = int(y)
    #     if surround:
    #         if x > 0:
    #             x, w = x - 1, w + 1
    #         else:
    #             x, w = x, w
    #         if y > 0:
    #             y, h = y - 1, h + 1
    #         else:
    #             y, h = y, h
    #     bottom = min(209, y + h)
    #     right = min(159, x + w)
    #     image_array[y:bottom + 1, x] = color
    #     image_array[y:bottom + 1, right] = color
    #     image_array[y, x:right + 1] = color
    #     image_array[bottom, x:right + 1] = color


    def _add_margin(self, pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result


    def _draw_objects_overlay(self, obs_image, action=None):
        obs_mod = deepcopy(obs_image)

        # Überprüfen, ob obs_image ein Bild oder ein Zustandsvektor ist
        if obs_mod.ndim == 1:  # Zustandsvektor
            print(f"Observation is a state vector, skipping object overlay. Shape: {obs_mod.shape}")
            return obs_mod  # Keine Modifikation für Zustandsvektoren

        if self.focus.ENV_NAME == "LunarLander-v5":
            objects = self.extract_all_objects()
            for obj in objects:
                mark_bb(obs_mod, obj.xywh, color=obj.rgb, name=str(obj))
        else:
            for obj in self.oc_env.objects:
                mark_bb(obs_mod, obj.xywh, color=obj.rgb, name=str(obj))
        return obs_mod



    def _draw_relation_overlay(self, obs_image, feature_vector, freeze_mask, action=None):
        scale = 4
        img = Image.fromarray(obs_image)

        # Überprüfen und Konvertieren des Bildmodus
        if img.mode != "RGBA":
            print(f"Converting image mode from {img.mode} to RGBA")
            img = img.convert("RGBA")

        draw = ImageDraw.Draw(img, "RGBA")

        if len(self.feature_attribution) == 0:
            img = img.resize((img.size[0] * scale, img.size[1] * scale), resample=Image.BOX)
            return np.array(img)

        features = self.feature_vector_description[0]
        fv_backmap = self.feature_vector_description[1]
        i = 0
        top_features_k = 5
        top_features_names = ["" for _ in range(top_features_k)]
        if np.ptp(self.feature_attribution):
            feature_attribution = (255 * (self.feature_attribution - np.min(self.feature_attribution)) / np.ptp(self.feature_attribution)).astype(int)
            top_features_idxs = np.argsort(feature_attribution)[-top_features_k:][::-1]
            for feature in features:
                i += 1
                idxs = np.where(fv_backmap == i - 1)[0]
                feature_name = feature[0]
                feature_signature = feature[1]
                fv_entries = feature_vector[idxs[0]:idxs[-1] + 1]
                fv_attribution = feature_attribution[idxs[0]:idxs[-1] + 1]
                fv_freeze_mask = freeze_mask[idxs[0]:idxs[-1] + 1]
                alpha = int(np.mean(fv_attribution) ** 2 / 255)
                # Implementieren Sie Ihre Zeichnungslogik hier
        img = img.resize((img.size[0] * scale, img.size[1] * scale), resample=Image.BOX)
        self._top_features = top_features_names
        return np.array(img)


    def get_vector_entry_descriptions(self):
        features = self.feature_vector_description[0]
        fv_backmap = self.feature_vector_description[1]
        i = 0
        features_names = []
        for feature in features:
            idxs = np.where(fv_backmap == i)[0]
            feature_name = feature[0]
            feature_signature = feature[1]
            for ii, idx in enumerate(idxs):
                features_names.append(format_feature(feature_name, feature_signature, ii))
            i += 1
        return features_names


def format_feature(feature_name, feature_signature, ii):
    if feature_name == 'RGB':
        axis = ["R", "G", "B"][ii]
        return f"RGB({feature_signature}.{axis})"
    if feature_name == "POSITION_HISTORY":
        if ii < 2:
            axis = ["x", "y"][ii]
            return f"{feature_signature}.{axis}"
        axis = ["x", "y"][ii-2]
        return f"{feature_signature}.{axis}[t-1]"
    axis = ["x", "y"][ii]
    if ii > 3:
        print("feature render formatting error. exiting...")
        exit()
    if feature_name == 'POSITION':
        return f"{feature_signature}.{axis}"
    elif feature_name == "EUCLIDEAN_DISTANCE":
        return f"ED({feature_signature[0][1]}, {feature_signature[1][1]})"
    elif feature_name == "DISTANCE":
        return f"D({feature_signature[0][1]}, {feature_signature[1][1]}).{axis}"
    elif feature_name == "VELOCITY":
        return f"V({feature_signature[0][1]}).{axis}"
    elif feature_name == "DIR_VELOCITY":
        return f"DV({feature_signature[0][1]}).{axis}"
    elif feature_name == "CENTER":
        return f"C({feature_signature[0][1]}, {feature_signature[1][1]}).{axis}"
    elif feature_name == "ORIENTATION":
        return f"O({feature_signature})"
    elif feature_name == "LINEAR_TRAJECTORY":
        return f"LT({feature_signature[0][1]}, {feature_signature[1][1]}).{axis}"
    elif feature_name == "COLOR":
        return f"COL({feature_signature[0][1]})"
    print("feature render formatting error. exiting...")
    exit()


def _make_darker(color, col_precent=0.8):
    """
    return a darker color
    """
    if not color:
        print("No color passed, using default black")
        return [0, 0, 0]
    return [int(col * col_precent) for col in color]


def mark_bb(image_array, bb, color=(255, 0, 0), surround=True, name=None):
    """
    Marks a bounding box on the image.
    """
    # Überprüfen, ob es sich um ein Bild handelt
    if image_array.ndim == 1:  # Zustandsvektor
        print(f"Cannot mark bounding box on state vector. Shape: {image_array.shape}")
        return

    x, y, w, h = bb
    x, y, w, h = int(x), int(y), int(w), int(h)
    color = _make_darker(color)

    if surround:
        if x > 0:
            x, w = x - 1, w + 1
        if y > 0:
            y, h = y - 1, h + 1

    y = min(209, y)
    x = min(159, x)
    bottom = min(209, y + h)
    right = min(159, x + w)

    print(f"Marking bounding box at: x={x}, y={y}, bottom={bottom}, right={right}, color={color}")

    bottom = int(bottom)
    right = int(right)

    image_array[y:bottom + 1, x] = color
    image_array[y:bottom + 1, right] = color
    image_array[y, x:right + 1] = color
    image_array[bottom, x:right + 1] = color


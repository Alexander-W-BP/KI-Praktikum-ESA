from termcolor import colored
import gymnasium as gym
# TODO: make nonstatic
OCATARI_AVAILABLE_GAMES = ["Boxing", "Skiing", "Pong","ALE/LunarLander-v5",]



# file to delegate to different object producing environments (wrappers)
def make(env_name, logger, notify=False):
    if notify:
        print("Env Name:", env_name)
    if env_name == "ALE/LunarLander-v5":
        env = gym.make("LunarLander-v2")
        logger.GeneralInfo("Environment %s specified. Compatible object extractor %s loaded." % (colored(env_name, "light_cyan"), colored("Gym", "light_cyan")))
        return env
    if True: # check if game is available and delegate
        import scobi.environments.ocgym as ocgym
        env = ocgym.make(env_name, notify)
        # TODO: get env name from OC_atari instance
        logger.GeneralInfo("Environment %s specified. Compatible object extractor %s loaded." % (colored(env_name, "light_cyan"),colored("OC_Atari", "light_cyan")))
        return env
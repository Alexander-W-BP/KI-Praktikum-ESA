from termcolor import colored
import gymnasium as gym

OCATARI_AVAILABLE_GAMES = ["ALE/Boxing-v5", "ALE/Skiing-v5", "ALE/Pong-v5", "ALE/LunarLander-v5"]

def make(env_name, logger, notify=False):
    if notify:
        print("Env Name:", env_name)
    if env_name == "ALE/LunarLander-v5":
        env = gym.make("LunarLander-v2")
        logger.GeneralInfo("Environment %s specified. Compatible object extractor %s loaded." % (colored(env_name, "light_cyan"), colored("Gym", "light_cyan")))
        return env
    elif env_name in OCATARI_AVAILABLE_GAMES:
        import scobi.environments.ocgym as ocgym
        env = ocgym.make(env_name, notify)
        logger.GeneralInfo("Environment %s specified. Compatible object extractor %s loaded." % (colored(env_name, "light_cyan"), colored("OC_Atari", "light_cyan")))
        return env
    else:
        raise ValueError(f"Environment {env_name} is not available.")
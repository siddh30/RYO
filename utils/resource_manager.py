from config import Config
conf = Config()


class ResourceManager(object):

    __instance = None

    def __init__(self):
        if ResourceManager.__instance is None:
            ResourceManager.__instance = self
        else:
            raise UserWarning('ResourceManager is a singleton. Use ResourceManager.get_instance() instead')

    @staticmethod
    def get_instance():
        if ResourceManager.__instance is None:
            ResourceManager()
        return ResourceManager.__instance

    @staticmethod
    def prompt_loader(prompt_name):
        with open(f"{conf.prompt_dir_path}/{prompt_name}.txt", "r", encoding="utf8") as f:
            return f.read()

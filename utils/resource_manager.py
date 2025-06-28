from config import Config
conf = Config()

class ResourceManager(object):

    __instance = None

    def __init__(self):
        if ResourceManager.__instance == None:
            ResourceManager.__instance = self
        
        else:
            raise UserWarning('ResourceManager is a singleton. User ResourceManager.get_instance() instead')

    @staticmethod
    def get_instance():
        if ResourceManager.__instance is None:
            ResourceManager()

        return ResourceManager.__instance
    

    ########## Load Prompts ##############
    @staticmethod
    def prompt_loader(prompt_name):
        with open(f"{conf.prompt_dir_path}/{prompt_name}.txt", "r", encoding="utf8") as f:
            prompt = f.read()
        
        return prompt


    ########## Stream Agents ##############
    @staticmethod
    def stream_agent(stream, show_stream=True):
        output = []
        for s in stream:
            message = s["messages"][-1]
            if show_stream==True:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
                
            output.append(message)

        return output

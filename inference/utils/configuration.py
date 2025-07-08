import json
import os


class Dict(dict):
    """
    Dictionary that allows to access per attributes and to except names from being loaded
    """
    def __init__(self, dictionary: dict = None):
        super(Dict, self).__init__()

        if dictionary is not None:
            self.load(dictionary)

    def __getattr__(self, item):
        try:
            return self[item] if item in self else getattr(super(Dict, self), item)
        except AttributeError:
            raise AttributeError(f'This dictionary has no attribute "{item}"')

    def load(self, dictionary: dict, name_list: list = None):
        """
        Loads a dictionary
        :param dictionary: Dictionary to be loaded
        :param name_list: List of names to be updated
        """
        for name in dictionary:
            data = dictionary[name]
            if name_list is None or name in name_list:
                if isinstance(data, dict):
                    if name in self:
                        self[name].load(data)
                    else:
                        self[name] = Dict(data)
                elif isinstance(data, list):
                    self[name] = list()
                    for item in data:
                        if isinstance(item, dict):
                            self[name].append(Dict(item))
                        else:
                            self[name].append(item)
                else:
                    self[name] = data

    def save(self, path):
        """
        Saves the dictionary into a json file
        :param path: Path of the json file
        """
        os.makedirs(path, exist_ok=True)

        path = os.path.join(path, 'cfg.json')

        with open(path, 'w') as file:
            json.dump(self, file, indent=True)


class Configuration(Dict):
    def __init__(self, path: str = None, json_string: str = None):
        super(Configuration, self).__init__()
        if path:
            self.base_path = os.path.dirname(path)  # Store the directory of the primary configuration file
            self.load_recursive(path)
        if json_string:
            self.load_json_string(json_string)

        # pretty print the configuration
        print(json.dumps(self, indent=4))

    def load_recursive(self, path: str):
        with open(path) as file:
            data = json.loads(file.read())
        if "parent_config" in data:
            parent_path = data["parent_config"]
            # Check if the parent path is just a filename
            if not os.path.dirname(parent_path):
                parent_path = os.path.join(self.base_path, parent_path)
            self.load_recursive(parent_path)
        super(Configuration, self).load(data)

    def load_json_string(self, json_string: str):
        data = json.loads(json_string)
        super(Configuration, self).load(data)

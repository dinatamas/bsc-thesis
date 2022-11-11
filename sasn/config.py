import importlib

# --------------------
# Settings management.
# --------------------

class SettingNotFoundError(Exception):
    def __init__(self, setting_name):
        super().__init__()
        self.setting_name = setting_name

    def __str__(self):
        return f"'{self.setting_name}' has not been set in the settings"

class Settings:
    def __getattr__(self, key):
        raise SettingNotFoundError(key)

    def configure(self, name):
        # Import the configuration module.
        module = importlib.import_module(f'datasets.{name}.settings')

        # Register the settings.
        for setting in dir(module):
            if setting.isupper():
                setattr(self, setting, getattr(module, setting))

    def __str__(self):
        strings = list()
        for setting in dir(self):
            if setting.isupper():
                strings.append(f'{setting} = {getattr(self, setting)}')
        return f'[{", ".join(strings)}]'

# -----------------------------------
# The module level settings instance.
# -----------------------------------

settings = Settings()

class Logger:
    enabled: bool = True
    
    @staticmethod
    def disable() -> None:
        Logger.enabled = False
        
    @staticmethod
    def enable() -> None:
        Logger.enabled = True
    
    @staticmethod
    def log(message: str) -> None:
        if Logger.enabled:
            print(f"[LOG]: {message}")
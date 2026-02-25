import logging
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class JSONFileHandler(logging.Handler):
    
    def __init__(self, filename, mode='a', encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self._ensure_path_exists()
        self._initialize_file()
    
    def _ensure_path_exists(self):
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
    
    def _initialize_file(self):
        path = Path(self.filename)
        if not path.exists() or path.stat().st_size == 0:
            with open(self.filename, 'w', encoding=self.encoding) as f:
                f.write('[]')
    
    def _sanitize_for_json(self, obj):
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)
    
    def _create_log_entry(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return self._sanitize_for_json(log_entry)
    
    def emit(self, record):
        try:
            log_entry = self._create_log_entry(record)
            
            # Read existing logs
            with open(self.filename, 'r', encoding=self.encoding) as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
                else:
                    logs = []
            
            # Append new log
            logs.append(log_entry)
            
            # Write back the entire array
            with open(self.filename, 'w', encoding=self.encoding) as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception:
            self.handleError(record)


def setup_logging():
    logger = logging.getLogger("Chatbot")
    logger.setLevel(logging.INFO)

    # create a file handler for JSON logs
    file_handler = JSONFileHandler("chatbot_logs.json")
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    )
    
    logger.addHandler(console_handler)

    return logger

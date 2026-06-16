import logging
import os
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler

class VLMMetricsLogger:
    
    def __init__(self, log_dir=None, log_file=None, max_bytes=10*1024*1024, backup_count=5):
        self.log_dir = log_dir or os.getenv('CONTAINER_RESULTS_PATH')
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_id = uuid.uuid4().hex[:6]        
        if log_file is None:            
            self.log_file = f"vlm_application_metrics_{timestamp}_{unique_id}.txt"
        else:
            self.log_file = log_file
        
        # Performance metrics file
        self.performance_log_file = f"vlm_performance_metrics_{timestamp}_{unique_id}.txt"
        
        self.logger = None
        self.performance_logger = None
        self._max_bytes = max_bytes
        self._backup_count = backup_count
    
    def _ensure_log_dir(self):
        """Create logs directory if it doesn't exist"""
        os.makedirs(self.log_dir, exist_ok=True)

    def _cleanup_existing_files(self, prefix):
        """Delete existing metrics files matching prefix"""
        if self.log_dir:
            for filename in os.listdir(self.log_dir):
                if filename.startswith(prefix):
                    file_path = os.path.join(self.log_dir, filename)
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

    def _ensure_app_logger(self):
        """Setup application logger on first use"""
        if self.logger is not None:
            return
        self._ensure_log_dir()
        self._cleanup_existing_files('vlm_application_metrics')
        
        self.logger = logging.getLogger('vlm_metrics_logger')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            log_path = os.path.join(self.log_dir, self.log_file)
            file_handler = RotatingFileHandler(
                log_path, 
                maxBytes=self._max_bytes, 
                backupCount=self._backup_count
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _ensure_performance_logger(self):
        """Setup performance logger on first use"""
        if self.performance_logger is not None:
            return
        self._ensure_log_dir()
        self._cleanup_existing_files('vlm_performance_metrics')
        
        self.performance_logger = logging.getLogger('vlm_performance_logger')
        self.performance_logger.setLevel(logging.INFO)
        
        if not self.performance_logger.handlers:
            performance_log_path = os.path.join(self.log_dir, self.performance_log_file)
            performance_file_handler = RotatingFileHandler(
                performance_log_path,
                maxBytes=self._max_bytes,
                backupCount=self._backup_count
            )
            performance_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            performance_file_handler.setFormatter(performance_formatter)
            self.performance_logger.addHandler(performance_file_handler)
    
    def log_start_time(self, usecase_name=None, unique_id='retail-default'):
        self._ensure_app_logger()
        timestamp_ms = int(time.time() * 1000)
        log_data = {
            'application': os.getenv(usecase_name),
            'id': unique_id,
            'event': 'start',
            'timestamp_ms': timestamp_ms
        }
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        self.logger.info(message)
        return timestamp_ms
    
    def user_log_start_time(self, timestamp_milliseconds, usecase_name=None, unique_id='retail-default'):
        self._ensure_app_logger()
        timestamp_ms = int(timestamp_milliseconds)
        log_data = {
            'application': os.getenv(usecase_name),
            'id': unique_id,
            'event': 'start',
            'timestamp_ms': timestamp_ms
        }
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        self.logger.info(message)
        return timestamp_ms
        
    def log_end_time(self, usecase_name, unique_id='retail-default'):
        self._ensure_app_logger()
        timestamp_ms = int(time.time() * 1000)
        log_data = {
            'application': os.getenv(usecase_name),
            'id': unique_id,
            'event': 'end',
            'timestamp_ms': timestamp_ms
        }
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        self.logger.info(message)
        return timestamp_ms
    
    def user_log_end_time(self, timestamp_milliseconds, usecase_name, unique_id='retail-default'):
        self._ensure_app_logger()
        timestamp_ms = int(timestamp_milliseconds)
        log_data = {
            'application': os.getenv(usecase_name),
            'id': unique_id,
            'event': 'end',
            'timestamp_ms': timestamp_ms
        }
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        self.logger.info(message)
        return timestamp_ms
        
    def log_custom_event(self, event_type, usecase_name, unique_id='retail-default', **kwargs):
        self._ensure_app_logger()
        timestamp_ms = int(time.time() * 1000)
        log_data = {
            'application': os.getenv(usecase_name),
            'id': unique_id,
            'event': event_type,
            'timestamp_ms': timestamp_ms
        }
        log_data.update(kwargs)
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        if event_type.lower() == 'error':
            self.logger.error(message)
        elif event_type.lower() == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
        return timestamp_ms
        
    def log_ovms_performance_metrics(self, usecase_name, vlm_metrics_result):
        self._ensure_performance_logger()
        timestamp_ms = int(time.time() * 1000)
        log_data = {
            'application': os.getenv(usecase_name),
            'timestamp_ms': timestamp_ms,
        }
        if isinstance(vlm_metrics_result, dict):
            log_data.update(vlm_metrics_result)
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        self.performance_logger.info(message)
        return timestamp_ms
        
    def log_performance_metrics(self, usecase_name, vlm_metrics_result_object, unique_id='retail-default'):
        self._ensure_performance_logger()
        timestamp_ms = int(time.time() * 1000)
        log_data = {
            'application':  os.getenv(usecase_name),
            'id': unique_id,
            'timestamp_ms': timestamp_ms,
            'Load_Time' : vlm_metrics_result_object.perf_metrics.get_load_time(),
            'Generated_Tokens':vlm_metrics_result_object.perf_metrics.get_num_generated_tokens(),
            'Input_Tokens':vlm_metrics_result_object.perf_metrics.get_num_input_tokens(),
            'TTFT_Mean':vlm_metrics_result_object.perf_metrics.get_ttft().mean,
            'TPOT_Mean':vlm_metrics_result_object.perf_metrics.get_tpot().mean,
            'Throughput_Mean':vlm_metrics_result_object.perf_metrics.get_throughput().mean,
            'Generate_Duration_Mean':vlm_metrics_result_object.perf_metrics.get_generate_duration().mean,
            'Tokenization_Duration_Mean':vlm_metrics_result_object.perf_metrics.get_tokenization_duration().mean,
            'Detokenization_Duration_Mean':vlm_metrics_result_object.perf_metrics.get_detokenization_duration().mean,
            'Grammar_Compile_Max':vlm_metrics_result_object.perf_metrics.get_grammar_compile_time().max,
            'Grammar_Compile_Min':vlm_metrics_result_object.perf_metrics.get_grammar_compile_time().min,
            'Grammar_Compile_Std':vlm_metrics_result_object.perf_metrics.get_grammar_compile_time().std,
            'Grammar_Compile_Mean':vlm_metrics_result_object.perf_metrics.get_grammar_compile_time().mean
        }
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        self.performance_logger.info(message)
        return timestamp_ms


# Global logger instance (singleton pattern)
_vlm_metrics_logger = None

def get_logger():
    """Get the global logger instance"""
    global _vlm_metrics_logger
    if _vlm_metrics_logger is None:
        _vlm_metrics_logger = VLMMetricsLogger()
    return _vlm_metrics_logger

def log_start_time(application_name, unique_id='retail-default'):
    """Convenience function for logging start time"""
    return get_logger().log_start_time(application_name, unique_id=unique_id)

def log_end_time(application_name, unique_id='retail-default'):
    """Convenience function for logging end time"""
    return get_logger().log_end_time(application_name, unique_id=unique_id)

def user_log_start_time(timestamp_milliseconds, application_name, unique_id='retail-default'):
    """Convenience function for logging start time"""
    return get_logger().user_log_start_time(timestamp_milliseconds, application_name, unique_id=unique_id)

def user_log_end_time(timestamp_milliseconds, application_name, unique_id='retail-default'):
    """Convenience function for logging end time"""
    return get_logger().user_log_end_time(timestamp_milliseconds, application_name, unique_id=unique_id)
    
def log_custom_event(event_type, application_name, unique_id='retail-default', **kwargs):
    """Convenience function for logging custom events"""
    return get_logger().log_custom_event(event_type, application_name, **kwargs)

def log_performance_metric(application_name,metrics, unique_id='retail-default'):
    """Convenience function for logging performance metrics"""
    return get_logger().log_performance_metrics(application_name,metrics, unique_id=unique_id)

def log_ovms_performance_metric(application_name,metrics):
    """Convenience function for logging OVMS performance metrics"""
    return get_logger().log_ovms_performance_metrics(application_name,metrics)

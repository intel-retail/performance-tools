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
        self._setup_logger(max_bytes, backup_count)
    
    def _setup_logger(self, max_bytes, backup_count):
        """Setup the logger with file rotation"""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Delete existing VLM metrics files if they exist
        if self.log_dir:
            for filename in os.listdir(self.log_dir):
                if filename.startswith('vlm_application_metrics') or filename.startswith('vlm_performance_metrics'):
                    file_path = os.path.join(self.log_dir, filename)
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass  # Ignore errors if file can't be deleted
        
        # Create main logger
        self.logger = logging.getLogger('vlm_metrics_logger')
        self.logger.setLevel(logging.INFO)
        
        # Create performance logger
        self.performance_logger = logging.getLogger('vlm_performance_logger')
        self.performance_logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Create file handler for main logs
            log_path = os.path.join(self.log_dir, self.log_file)
            file_handler = RotatingFileHandler(
                log_path, 
                maxBytes=max_bytes, 
                backupCount=backup_count
            )
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Set formatter
            file_handler.setFormatter(formatter)
            
            # Add handler to main logger
            self.logger.addHandler(file_handler)
        
        # Setup performance logger
        if not self.performance_logger.handlers:
            # Create file handler for performance logs
            performance_log_path = os.path.join(self.log_dir, self.performance_log_file)
            performance_file_handler = RotatingFileHandler(
                performance_log_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            
            # Create formatter for performance logs
            performance_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            performance_file_handler.setFormatter(performance_formatter)
            
            # Add handler to performance logger (no console output for performance)
            self.performance_logger.addHandler(performance_file_handler)
    
    def log_start_time(self, usecase_name):
        
        timestamp_ms = int(time.time() * 1000)
        
        log_data = {
            'application': os.getenv(usecase_name),
            'event': 'start',
            'timestamp_ms': timestamp_ms
        }
        
        # Format log message
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        
        self.logger.info(message)
        return timestamp_ms
    
    def log_end_time(self, usecase_name):
        
        timestamp_ms = int(time.time() * 1000)
        
        log_data = {
            'application': os.getenv(usecase_name),
            'event': 'end',
            'timestamp_ms': timestamp_ms
        }
            
        # Format log message
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        
        self.logger.info(message)
        return timestamp_ms
    
    def log_custom_event(self, event_type, usecase_name, **kwargs):
        timestamp_ms = int(time.time() * 1000)
        
        log_data = {
            'application': os.getenv(usecase_name),
            'event': event_type,
            'timestamp_ms': timestamp_ms
        }
        
        # Add custom parameters
        log_data.update(kwargs)
        
        # Format log message
        message_parts = [f"{key}={value}" for key, value in log_data.items()]
        message = " ".join(message_parts)
        
        # Log at appropriate level
        if event_type.lower() == 'error':
            self.logger.error(message)
        elif event_type.lower() == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        return timestamp_ms
    
    def log_performance_metrics(self, usecase_name, vlm_metrics_result_object):
        
        timestamp_ms = int(time.time() * 1000)
        log_data = {
            'application':  os.getenv(usecase_name),
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
        
                
        # Format log message
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

def log_start_time(application_name):
    """Convenience function for logging start time"""
    return get_logger().log_start_time(application_name)

def log_end_time(application_name):
    """Convenience function for logging end time"""
    return get_logger().log_end_time(application_name)

def log_custom_event(event_type, application_name, **kwargs):
    """Convenience function for logging custom events"""
    return get_logger().log_custom_event(event_type, application_name, **kwargs)

def log_performance_metric(application_name,metrics):
    """Convenience function for logging performance metrics"""
    return get_logger().log_performance_metrics(application_name,metrics)

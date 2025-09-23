#!/usr/bin/env python3
"""
sofia_auto_startup.py
Sistema de auto-inicio para SOFIA AGI
Permite que SOFIA se inicie automáticamente cuando detecta necesidad
"""
import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Configurar logging
logging.basicConfig(
    filename='./SOFIA/auto_startup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SOFIAAutoStartup:
    """Sistema de auto-inicio para SOFIA"""

    def __init__(self):
        self.config_file = './config.yaml'
        self.pid_file = './SOFIA/sofia.pid'
        self.status_file = './SOFIA/sofia_status.json'
        self.is_running = False
        self.startup_triggers = self._load_startup_triggers()

    def _load_startup_triggers(self) -> Dict[str, Any]:
        """Carga los triggers de auto-inicio"""
        try:
            with open(self.config_file, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
                growth_config = config.get('model_growth', {})
                return growth_config.get('expansion_triggers', {})
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}

    def check_startup_conditions(self) -> bool:
        """Verifica si se deben cumplir las condiciones de inicio"""
        triggers = self.startup_triggers

        # Trigger: interacción del usuario
        if triggers.get('user_demand', False):
            if self._detect_user_activity():
                logging.info("User activity detected - triggering startup")
                return True

        # Trigger: hora del día
        if self._is_optimal_startup_time():
            logging.info("Optimal startup time reached")
            return True

        # Trigger: recursos disponibles
        if self._resources_available():
            logging.info("Resources available for startup")
            return True

        # Trigger: mantenimiento programado
        if self._maintenance_due():
            logging.info("Maintenance due - triggering startup")
            return True

        # Trigger: training pendiente
        if self._training_checkpoint_available():
            logging.info("Training checkpoint available - triggering startup")
            return True

        return False

    def _detect_user_activity(self) -> bool:
        """Detecta actividad del usuario"""
        try:
            # Verificar procesos de usuario
            user_processes = ['code', 'chrome', 'firefox', 'terminal', 'konsole', 'gnome-terminal']
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and any(up in proc.info['name'].lower() for up in user_processes):
                    return True
        except:
            pass
        return False

    def _is_optimal_startup_time(self) -> bool:
        """Verifica si es hora óptima de inicio"""
        now = datetime.now()
        # Iniciar entre 9 AM y 6 PM
        return 9 <= now.hour <= 18

    def _resources_available(self) -> bool:
        """Verifica si hay recursos disponibles"""
        try:
            # Verificar CPU (< 50% uso)
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 50:
                return False

            # Verificar memoria (> 2GB libre)
            memory = psutil.virtual_memory()
            free_gb = memory.available / (1024**3)
            if free_gb < 2:
                return False

            # Verificar GPU si está disponible
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    if gpu.memoryUsed / gpu.memoryTotal > 0.8:  # > 80% GPU memory used
                        return False
            except:
                pass

            return True
        except:
            return False

    def _maintenance_due(self) -> bool:
        """Verifica si el mantenimiento está pendiente"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    status = json.load(f)

                last_maintenance = status.get('last_maintenance')
                if last_maintenance:
                    last_mt = datetime.fromisoformat(last_maintenance)
                    # Mantenimiento cada 24 horas
                    return datetime.now() - last_mt > timedelta(hours=24)
        except:
            pass
        return True  # Si no hay registro, hacer mantenimiento

    def _training_checkpoint_available(self) -> bool:
        """Verifica si hay un checkpoint de training disponible para reanudar"""
        try:
            # Verificar si existe directorio SOFIA-AGI (donde se guarda el training)
            training_dir = './SOFIA-AGI'
            if not os.path.exists(training_dir):
                return False

            # Verificar si hay archivos de checkpoint
            checkpoint_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json']
            has_checkpoint = any(os.path.exists(os.path.join(training_dir, f)) for f in checkpoint_files)

            if has_checkpoint:
                logging.info("Training checkpoint found in SOFIA-AGI directory")
                return True

        except Exception as e:
            logging.error(f"Error checking training checkpoint: {e}")

        return False

    def is_sofia_running(self) -> bool:
        """Verifica si SOFIA ya está ejecutándose"""
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())

                # Verificar si el proceso existe
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    # Verificar si es un proceso de SOFIA
                    if 'sofia' in ' '.join(proc.cmdline()).lower():
                        return True
            except:
                pass

            # Limpiar PID file si el proceso no existe
            os.remove(self.pid_file)

        return False

    def start_sofia(self) -> bool:
        """Inicia SOFIA"""
        if self.is_sofia_running():
            logging.info("SOFIA already running")
            return False

        try:
            logging.info("Starting SOFIA...")

            # Cambiar al directorio de SOFIA
            os.chdir(os.path.dirname(os.path.abspath(__file__)))

            # Iniciar SOFIA daemon en background
            process = subprocess.Popen(
                [sys.executable, 'sofia_daemon.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )

            # Guardar PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))

            # Actualizar status
            self._update_status('running', process.pid)

            logging.info(f"SOFIA daemon started with PID: {process.pid}")

            # Si hay training pendiente, iniciarlo también
            if self._training_checkpoint_available():
                logging.info("Starting training resumption...")
                self._resume_training()

            return True

        except Exception as e:
            logging.error(f"Failed to start SOFIA: {e}")
            return False

    def stop_sofia(self) -> bool:
        """Detiene SOFIA"""
        if not self.is_sofia_running():
            logging.info("SOFIA not running")
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Enviar señal de terminación
            os.kill(pid, 15)  # SIGTERM

            # Esperar a que termine
            time.sleep(5)

            # Forzar terminación si aún está vivo
            if psutil.pid_exists(pid):
                os.kill(pid, 9)  # SIGKILL

            # Limpiar archivos
            os.remove(self.pid_file)
            self._update_status('stopped')

            logging.info("SOFIA stopped successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to stop SOFIA: {e}")
            return False

    def _update_status(self, status: str, pid: Optional[int] = None):
        """Actualiza el archivo de status"""
        try:
            status_data = {
                'status': status,
                'pid': pid,
                'timestamp': datetime.now().isoformat(),
                'last_maintenance': datetime.now().isoformat()
            }

            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)

        except Exception as e:
            logging.error(f"Failed to update status: {e}")

    def run_maintenance(self):
        """Ejecuta mantenimiento de SOFIA"""
        logging.info("Running SOFIA maintenance...")

        try:
            # Aquí iría la lógica de mantenimiento:
            # - Limpiar caches
            # - Optimizar modelo
            # - Actualizar conocimientos
            # - Verificar integridad

            logging.info("Maintenance completed")
            self._update_status('maintenance_completed')

    def _resume_training(self):
        """Reanuda el training desde checkpoint"""
        try:
            logging.info("Resuming AGI training from checkpoint...")

            # Ejecutar comando de training en background
            training_cmd = [
                sys.executable, 'train_sofia.py',
                '--config', 'config.yaml',
                '--train', './data/pairs_augmented.jsonl',
                '--out', './SOFIA-AGI',
                '--agi-training'
            ]

            # Iniciar training en background
            training_process = subprocess.Popen(
                training_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )

            logging.info(f"Training resumed with PID: {training_process.pid}")

        except Exception as e:
            logging.error(f"Failed to resume training: {e}")

    def run_monitor_loop(self):
        """Ejecuta el loop de monitoreo continuo"""
        logging.info("SOFIA Auto-Startup System")
        logging.info("Running in monitor mode...")

        while True:
            try:
                if self.check_startup_conditions():
                    if not self.is_sofia_running():
                        self.start_sofia()
                    else:
                        logging.info("SOFIA already running, skipping startup")

                # Verificar training cada hora
                if self._should_check_training():
                    if self._training_checkpoint_available() and not self._is_training_running():
                        logging.info("Resuming training automatically...")
                        self._resume_training()

                time.sleep(300)  # Verificar cada 5 minutos

            except KeyboardInterrupt:
                logging.info("Auto-startup monitor stopped by user")
                break
            except Exception as e:
                logging.error(f"Monitor loop error: {e}")
                time.sleep(60)  # Esperar 1 minuto antes de reintentar

    def _should_check_training(self) -> bool:
        """Verifica si debe chequear training (cada hora)"""
        # Implementar lógica para verificar cada hora
        return True  # Por ahora, siempre verificar

    def _is_training_running(self) -> bool:
        """Verifica si el training ya está ejecutándose"""
        try:
            for proc in psutil.process_iter(['cmdline']):
                if proc.info['cmdline'] and any('train_sofia.py' in str(cmd) for cmd in proc.info['cmdline']):
                    return True
        except:
            pass
        return False
        logging.info("Starting SOFIA auto-startup monitor")

        while True:
            try:
                # Verificar condiciones de inicio
                if self.check_startup_conditions():
                    if not self.is_sofia_running():
                        self.start_sofia()
                    else:
                        # Verificar si necesita mantenimiento
                        if self._maintenance_due():
                            self.run_maintenance()

                # Verificar si SOFIA sigue viva
                if self.is_sofia_running():
                    self._update_status('running')
                else:
                    self._update_status('stopped')

                # Dormir por un tiempo
                time.sleep(300)  # Revisar cada 5 minutos

            except Exception as e:
                logging.error(f"Monitor loop error: {e}")
                time.sleep(60)  # Esperar 1 minuto en caso de error

def main():
    """Función principal"""
    startup_system = SOFIAAutoStartup()

    # Parsear argumentos de línea de comandos
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'start':
            if startup_system.start_sofia():
                print("✅ SOFIA started successfully")
            else:
                print("❌ Failed to start SOFIA")

        elif command == 'stop':
            if startup_system.stop_sofia():
                print("✅ SOFIA stopped successfully")
            else:
                print("❌ Failed to stop SOFIA")

        elif command == 'status':
            status = "running" if startup_system.is_sofia_running() else "stopped"
            print(f"SOFIA status: {status}")

        elif command == 'monitor':
            print("Starting SOFIA monitor (press Ctrl+C to stop)...")
            try:
                startup_system.run_monitor_loop()
            except KeyboardInterrupt:
                print("\nMonitor stopped")

        else:
            print("Usage: python sofia_auto_startup.py [start|stop|status|monitor]")

    else:
        # Modo monitor por defecto
        print("SOFIA Auto-Startup System")
        print("Running in monitor mode...")
        try:
            startup_system.run_monitor_loop()
        except KeyboardInterrupt:
            print("\nMonitor stopped")

if __name__ == "__main__":
    main()

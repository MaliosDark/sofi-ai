#!/usr/bin/env python3
"""
sofia_daemon.py
Daemon principal de SOFIA AGI
Ejecuta SOFIA en background con crecimiento continuo
"""
import os
import sys
import time
import signal
import logging
import json
from datetime import datetime
import yaml
import torch
from typing import Dict, Any

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sofia_growth_system import SOFIAGrowthSystem
from sofia_llm_integration import SOFIALanguageModel

# Configurar logging
logging.basicConfig(
    filename='./SOFIA/sofia_daemon.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SOFIADaemon:
    """Daemon principal de SOFIA"""

    def __init__(self):
        self.config = self._load_config()
        self.growth_system = None
        self.llm = None
        self.running = True
        self.interaction_count = 0
        self.start_time = time.time()

        # Configurar signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logging.info("SOFIA Daemon initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración"""
        try:
            with open('./config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return {}

    def _signal_handler(self, signum, frame):
        """Maneja señales de terminación"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def initialize_systems(self):
        """Inicializa todos los sistemas de SOFIA"""
        try:
            logging.info("Initializing SOFIA systems...")

            # Inicializar sistema de crecimiento
            self.growth_system = SOFIAGrowthSystem(self.config)
            logging.info("✅ Growth system initialized")

            # Sincronizar estado del daemon con el sistema de crecimiento
            self.sync_with_growth_system()
            logging.info("✅ Daemon state synced with growth system")

            # Inicializar LLM
            self.llm = SOFIALanguageModel()
            logging.info("✅ LLM initialized")

            # Verificar GPU
            if torch.cuda.is_available():
                logging.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            else:
                logging.info("⚠️  No GPU available, using CPU")

            logging.info("🎉 SOFIA fully initialized and ready")

        except Exception as e:
            logging.error(f"Failed to initialize systems: {e}")
            raise

    async def process_interaction(self, user_input: str) -> str:
        """Procesa una interacción del usuario"""
        try:
            self.interaction_count += 1

            # Generar respuesta usando LLM
            if self.llm:
                response = await self.llm.generate_response(user_input)
            else:
                response = "Lo siento, mi sistema de lenguaje no está disponible."

            # Calcular calidad de la interacción
            interaction_quality = self._calculate_interaction_quality(user_input, response)

            # Registrar interacción en el sistema de crecimiento
            if self.growth_system:
                self.growth_system.record_interaction(interaction_quality)

            # Aprender de la interacción
            self._learn_from_interaction(user_input, response)

            return response

        except Exception as e:
            logging.error(f"Error processing interaction: {e}")
            return "Disculpa, tuve un problema procesando tu mensaje."

    def _calculate_interaction_quality(self, user_input: str, response: str) -> float:
        """Calcula la calidad de una interacción"""
        try:
            # Calidad básica basada en longitud y contenido
            quality = 0.5  # Base

            # Bonos por longitud razonable
            if 10 <= len(response) <= 500:
                quality += 0.2

            # Bonos por contenido específico
            if any(word in response.lower() for word in ['ayudar', 'entender', 'explicar', 'puedo']):
                quality += 0.1

            # Penalizaciones por respuestas genéricas
            if response.lower() in ['sí', 'no', 'ok', 'bien']:
                quality -= 0.2

            # Penalizaciones por errores
            if 'error' in response.lower() or 'problema' in response.lower():
                quality -= 0.3

            return max(0.0, min(1.0, quality))

        except Exception:
            return 0.5  # Calidad neutral por defecto

    def _learn_from_interaction(self, user_input: str, response: str):
        """Aprende de la interacción"""
        try:
            # Aquí iría la lógica de aprendizaje continuo
            # Por ahora, solo registramos
            logging.info(f"Learned from interaction #{self.interaction_count}")

            # Verificar si es momento de crecer
            if self.growth_system and self.interaction_count % 10 == 0:  # Cada 10 interacciones
                growth_status = self.growth_system.get_growth_status()
                logging.info(f"Growth status: Phase {growth_status['current_phase']}, "
                           f"Capability score: {growth_status['metrics']['capability_score']:.1f}")

        except Exception as e:
            logging.error(f"Error in learning: {e}")

    def sync_with_growth_system(self):
        """Sincroniza el estado del daemon con el sistema de crecimiento"""
        try:
            if self.growth_system:
                # Sincronizar contador de interacciones
                growth_metrics = self.growth_system.metrics
                self.interaction_count = growth_metrics.interaction_count
                
                # Actualizar estado del daemon
                self._update_daemon_status()
                
                logging.info(f"Synced daemon state: interactions={self.interaction_count}")
        except Exception as e:
            logging.error(f"Error syncing with growth system: {e}")

    def _update_daemon_status(self):
        """Actualiza el archivo de estado del daemon"""
        try:
            status = {
                "running": self.running,
                "interaction_count": self.interaction_count,
                "growth_status": {
                    "current_phase": self.growth_system.current_phase if self.growth_system else "unknown",
                    "metrics": self.growth_system.metrics.__dict__ if self.growth_system else {}
                },
                "llm_available": self.llm is not None,
                "gpu_available": torch.cuda.is_available(),
                "uptime": int(time.time() - self.start_time)
            }
            
            with open('./SOFIA/daemon_status.json', 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error updating daemon status: {e}")

    def perform_maintenance(self):
        """Realiza mantenimiento del sistema"""
        try:
            logging.info("Performing system maintenance...")

            # Limpiar cachés de PyTorch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU cache cleared")

            # Optimizar modelo si es necesario
            # Aquí iría lógica de optimización

            # Sincronizar con sistema de crecimiento
            self.sync_with_growth_system()

            logging.info("Maintenance completed")

        except Exception as e:
            logging.error(f"Maintenance error: {e}")

    def run_daemon_loop(self):
        """Loop principal del daemon"""
        logging.info("🎯 Starting SOFIA daemon loop")
        print("🎯 Daemon loop started - watching for growth opportunities...")

        # Crear archivo de estado inicial
        self._update_status_file()
        logging.info("📄 Initial status file created")

        maintenance_interval = 3600  # 1 hora
        last_maintenance = time.time()
        loop_count = 0

        while self.running:
            loop_count += 1

            try:
                current_time = time.time()

                # Realizar mantenimiento periódico
                if current_time - last_maintenance > maintenance_interval:
                    logging.info("🧹 Performing maintenance...")
                    self.perform_maintenance()
                    last_maintenance = current_time

                # Verificar crecimiento
                if self.growth_system:
                    growth_status = self.growth_system.get_growth_status()
                    if growth_status.get('is_growing'):
                        logging.info("SOFIA is currently growing...")
                        print("🌱 SOFIA is actively growing and expanding capabilities...")

                # Mostrar status cada 6 loops (1 minuto)
                if loop_count % 6 == 0:
                    uptime = current_time - self.start_time
                    phase = growth_status.get('current_phase', 'foundation') if self.growth_system else 'unknown'
                    print(f"📊 Status: Running for {uptime:.0f}s | Loops: {loop_count} | Phase: {phase} | Interactions: {self.interaction_count}")

                # Actualizar archivo de estado
                self._update_status_file()

                # Pequeña pausa para no consumir CPU
                time.sleep(10)

            except Exception as e:
                logging.error(f"Daemon loop error: {e}")
                print(f"❌ Error in daemon loop: {e}")
                time.sleep(60)  # Esperar 1 minuto en caso de error
                time.sleep(60)  # Esperar 1 minuto en caso de error

        logging.info("SOFIA daemon shutting down")

    def _update_status_file(self):
        """Actualiza el archivo de estado del daemon"""
        try:
            status = self.get_status()
            status_file = os.path.join(os.path.dirname(__file__), 'SOFIA', 'daemon_status.json')

            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)

        except Exception as e:
            logging.error(f"Error updating status file: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Obtiene el status del daemon"""
        return {
            'running': self.running,
            'interaction_count': self.interaction_count,
            'growth_status': self.growth_system.get_growth_status() if self.growth_system else None,
            'llm_available': self.llm is not None,
            'gpu_available': torch.cuda.is_available(),
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
        return {
            'running': self.running,
            'interaction_count': self.interaction_count,
            'growth_status': self.growth_system.get_growth_status() if self.growth_system else None,
            'llm_available': self.llm is not None,
            'gpu_available': torch.cuda.is_available(),
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }

def main():
    """Función principal del daemon"""
    logging.info("🚀 Starting SOFIA daemon main function")
    daemon = SOFIADaemon()
    logging.info("✅ SOFIADaemon instance created")

    try:
        logging.info("🔧 Initializing SOFIA systems...")
        # Inicializar sistemas
        daemon.initialize_systems()
        logging.info("✅ Systems initialized successfully")

        logging.info("🔄 Starting daemon loop...")
        # Ejecutar loop del daemon
        daemon.run_daemon_loop()
        logging.info("✅ Daemon loop completed")

    except Exception as e:
        logging.error(f"❌ Daemon failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

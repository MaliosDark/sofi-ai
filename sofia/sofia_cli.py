#!/usr/bin/env python3
"""
sofia_cli.py
Interfaz de línea de comandos para interactuar con SOFIA AGI
"""
import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from typing import Optional

class SOFIACLI:
    """Interfaz de línea de comandos para SOFIA"""

    def __init__(self):
        self.api_url = "http://localhost:8000"  # API endpoint cuando esté disponible
        self.daemon_status = self._check_daemon_status()

    def _check_daemon_status(self) -> dict:
        """Verifica el status del daemon de SOFIA"""
        try:
            # Leer archivo de status del daemon
            status_file = os.path.join(os.path.dirname(__file__), 'SOFIA', 'daemon_status.json')
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading daemon status: {e}")
        return {'running': False, 'status': 'unknown'}

    def show_welcome(self):
        """Muestra mensaje de bienvenida"""
        status = "RUNNING" if self.daemon_status.get('running', False) else "STOPPED"
        print("🤖 SOFIA AGI - Autonomous Growth Intelligence")
        print("=" * 50)
        print(f"Status: {status}")
        print(f"Phase: {self._get_growth_phase()}")
        print(f"Interactions: {self._get_interaction_count()}")
        print("=" * 50)
        print("Commands:")
        print("  /status    - Show SOFIA status")
        print("  /grow      - Force growth")
        print("  /train     - Start training")
        print("  /chat      - Start chat mode")
        print("  /quit      - Exit")
        print("=" * 50)

    def _get_growth_phase(self) -> str:
        """Obtiene la fase actual de crecimiento"""
        try:
            # Buscar en el directorio raíz primero, luego en sofia/
            for path in ['../SOFIA/growth_state.json', './SOFIA/growth_state.json']:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        state = json.load(f)
                        return state.get('current_phase', 'unknown')
        except:
            pass
        return 'unknown'

    def _get_interaction_count(self) -> int:
        """Obtiene el conteo de interacciones"""
        return self.daemon_status.get('interaction_count', 0)

    def show_status(self):
        """Muestra el status detallado de SOFIA"""
        print("\n📊 SOFIA STATUS")
        print("-" * 30)

        # Status del daemon
        daemon_status = "RUNNING" if self.daemon_status.get('running', False) else "STOPPED"
        print(f"Daemon Status: {daemon_status}")

        if self.daemon_status.get('running'):
            uptime = self.daemon_status.get('uptime', 0)
            print(f"Uptime: {uptime:.0f} seconds")
            # Get interactions from growth system instead of daemon
            interactions = self._get_interaction_count()
            print(f"Interactions: {interactions}")
            if self.daemon_status.get('llm_available'):
                print("LLM: Available")
            if self.daemon_status.get('gpu_available'):
                print("GPU: Available")

        # Información de crecimiento
        try:
            growth_state = None
            # Buscar en el directorio raíz primero, luego en sofia/
            for path in ['../SOFIA/growth_state.json', './SOFIA/growth_state.json']:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        growth_state = json.load(f)
                    break
            
            if growth_state:
                metrics = growth_state.get('metrics', {})
                print(f"Current Phase: {growth_state.get('current_phase', 'unknown')}")
                print(f"Capability Score: {metrics.get('capability_score', 0):.1f}/100")
                print(f"Autonomy Level: {metrics.get('autonomy_level', 0):.1f}/100")
                print(f"Knowledge Volume: {metrics.get('knowledge_volume', 0)} KB")
                print(f"Interactions: {metrics.get('interaction_count', 0)}")

                # Fases completadas
                phases = growth_state.get('growth_phases', {})
                completed = [name for name, phase in phases.items() if phase.get('completed', False)]
                print(f"Completed Phases: {', '.join(completed) if completed else 'None'}")
            else:
                print("Growth system: Not initialized")
        except Exception as e:
            print(f"Error reading growth state: {e}")

        print()

    def force_growth(self):
        """Fuerza el crecimiento de SOFIA"""
        try:
            from sofia_growth_system import SOFIAGrowthSystem
            import yaml

            # Cargar config
            with open('./config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Crear sistema de crecimiento
            growth_system = SOFIAGrowthSystem(config)

            print("🚀 Forcing SOFIA growth...")
            growth_system.force_growth()

            print("✅ Growth initiated")

        except Exception as e:
            print(f"❌ Failed to force growth: {e}")

    def start_training(self):
        """Inicia el entrenamiento de SOFIA"""
        print("🎓 Starting SOFIA training...")
        try:
            os.system("python train_sofia.py --config config.yaml --agi-training")
        except Exception as e:
            print(f"❌ Training failed: {e}")

    async def start_chat(self):
        """Inicia modo chat con SOFIA"""
        print("💬 Starting chat with SOFIA...")
        print("Type 'quit' to exit chat mode")
        print("-" * 40)

        # Aquí iría la lógica de chat
        # Por ahora, simular
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # Simular respuesta de SOFIA
            response = await self._get_sofia_response(user_input)
            print(f"SOFIA: {response}")

        print("Chat ended.")

    async def _get_sofia_response(self, user_input: str) -> str:
        """Obtiene respuesta de SOFIA"""
        # Aquí conectaría con el daemon o API
        # Por ahora, respuesta simulada
        return f"Entiendo que dijiste: '{user_input}'. Estoy aprendiendo y creciendo. Mi fase actual es {self._get_growth_phase()}."

    def run(self):
        """Ejecuta la interfaz CLI"""
        self.show_welcome()

        while True:
            try:
                command = input("sofia> ").strip().lower()

                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif command in ['status', 's']:
                    self.show_status()
                elif command in ['grow', 'g']:
                    self.force_growth()
                elif command in ['train', 't']:
                    self.start_training()
                elif command in ['chat', 'c']:
                    asyncio.run(self.start_chat())
                elif command in ['help', 'h', '?']:
                    self.show_welcome()
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='SOFIA AGI Command Line Interface')
    parser.add_argument('command', nargs='?', help='Command to execute')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')

    args = parser.parse_args()

    cli = SOFIACLI()

    if args.command:
        # Execute single command and exit
        if args.command == 'status':
            cli.show_status()
        elif args.command == 'grow':
            cli.force_growth()
        elif args.command == 'train':
            cli.start_training()
        elif args.command == 'chat':
            asyncio.run(cli.start_chat())
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    else:
        # Interactive mode
        cli.run()

if __name__ == "__main__":
    main()

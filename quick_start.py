#!/usr/bin/env python
"""
Quick start script for the fraud analytics platform.
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Fraud Analytics Platform Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('manage.py'):
        print("❌ manage.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Set up Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fraud_platform.settings')
    
    # Run migrations
    if not run_command("python manage.py makemigrations", "Creating migrations"):
        return False
    
    if not run_command("python manage.py migrate", "Running migrations"):
        return False
    
    # Create superuser (optional)
    print("\n👤 Would you like to create a superuser? (y/n): ", end="")
    if input().lower().startswith('y'):
        run_command("python manage.py createsuperuser", "Creating superuser")
    
    print("\n🎉 Setup complete!")
    print("\nTo start the development server, run:")
    print("python manage.py runserver")
    print("\nAPI will be available at: http://localhost:8000/")
    print("API Documentation: http://localhost:8000/docs/")
    print("Admin Panel: http://localhost:8000/admin/")

if __name__ == "__main__":
    main()
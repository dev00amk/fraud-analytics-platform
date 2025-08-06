#!/bin/bash

# 🚀 FREE Backup System for Fraud Analytics Platform
# Uses only free tools and services - $0 cost!

set -e

# Configuration
DB_NAME="fraud_platform"
DB_USER="postgres"
DB_HOST="localhost"
DB_PORT="5432"
BACKUP_DIR="/var/backups/fraud-analytics"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Free cloud storage options (pick one)
CLOUD_STORAGE="rclone"  # Options: rclone, aws-cli, gsutil, azure-cli

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory..."
    sudo mkdir -p "$BACKUP_DIR"
    sudo chown $(whoami):$(whoami) "$BACKUP_DIR"
}

# Database backup using pg_dump (FREE)
backup_database() {
    log "Starting database backup..."
    
    BACKUP_FILE="$BACKUP_DIR/db_backup_$TIMESTAMP.sql"
    
    # Full database backup
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"
    
    # Compress backup to save space
    gzip "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE}.gz"
    
    log "Database backup completed: $BACKUP_FILE"
    echo "$BACKUP_FILE"
}

# Application files backup
backup_application() {
    log "Starting application files backup..."
    
    APP_BACKUP_FILE="$BACKUP_DIR/app_backup_$TIMESTAMP.tar.gz"
    
    # Exclude unnecessary files and directories
    tar -czf "$APP_BACKUP_FILE" \
        --exclude='*/venv/*' \
        --exclude='*/node_modules/*' \
        --exclude='*/__pycache__/*' \
        --exclude='*/logs/*' \
        --exclude='*/.git/*' \
        --exclude='*/media/temp/*' \
        /path/to/fraud-analytics-platform/
    
    log "Application backup completed: $APP_BACKUP_FILE"
    echo "$APP_BACKUP_FILE"
}

# Redis backup (if using Redis)
backup_redis() {
    log "Starting Redis backup..."
    
    REDIS_BACKUP_FILE="$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"
    
    # Copy Redis dump file
    if [ -f "/var/lib/redis/dump.rdb" ]; then
        cp /var/lib/redis/dump.rdb "$REDIS_BACKUP_FILE"
        gzip "$REDIS_BACKUP_FILE"
        log "Redis backup completed: ${REDIS_BACKUP_FILE}.gz"
        echo "${REDIS_BACKUP_FILE}.gz"
    else
        warn "Redis dump file not found, skipping Redis backup"
    fi
}

# Upload to free cloud storage
upload_to_cloud() {
    local backup_file=$1
    
    case $CLOUD_STORAGE in
        "rclone")
            # Rclone supports many free cloud services
            # Setup: rclone config (supports Google Drive, OneDrive, Dropbox, etc.)
            log "Uploading to cloud storage via rclone..."
            rclone copy "$backup_file" "remote:fraud-analytics-backups/"
            ;;
        "aws-cli")
            # AWS Free Tier: 5GB S3 storage
            log "Uploading to AWS S3..."
            aws s3 cp "$backup_file" "s3://your-backup-bucket/"
            ;;
        "gsutil")
            # Google Cloud Free Tier: 5GB storage
            log "Uploading to Google Cloud Storage..."
            gsutil cp "$backup_file" "gs://your-backup-bucket/"
            ;;
        "azure-cli")
            # Azure Free Tier: 5GB storage
            log "Uploading to Azure Blob Storage..."
            az storage blob upload --file "$backup_file" --name "$(basename $backup_file)" --container-name backups
            ;;
        *)
            warn "No cloud storage configured"
            ;;
    esac
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups (older than $RETENTION_DAYS days)..."
    
    find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "*.rdb.gz" -mtime +$RETENTION_DAYS -delete
    
    log "Cleanup completed"
}

# Send notification (FREE options)
send_notification() {
    local status=$1
    local message=$2
    
    # Email notification using sendmail (FREE)
    if command -v sendmail &> /dev/null; then
        echo "Subject: Fraud Analytics Backup $status
        
        $message
        
        Timestamp: $(date)
        Host: $(hostname)" | sendmail admin@yourdomain.com
    fi
    
    # Slack webhook notification (FREE)
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🔄 Fraud Analytics Backup $status: $message\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
    
    # Discord webhook notification (FREE)
    if [ ! -z "$DISCORD_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"content\":\"🔄 Fraud Analytics Backup $status: $message\"}" \
            "$DISCORD_WEBHOOK_URL"
    fi
}

# Health check before backup
health_check() {
    log "Performing health check..."
    
    # Check database connectivity
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; then
        error "Database is not accessible"
    fi
    
    # Check disk space (need at least 1GB free)
    available_space=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 1048576 ]; then  # 1GB in KB
        error "Insufficient disk space for backup"
    fi
    
    log "Health check passed"
}

# Main backup function
main() {
    log "🚀 Starting FREE Backup System for Fraud Analytics Platform"
    
    # Pre-backup checks
    health_check
    create_backup_dir
    
    # Perform backups
    DB_BACKUP=$(backup_database)
    APP_BACKUP=$(backup_application)
    REDIS_BACKUP=$(backup_redis)
    
    # Upload to cloud storage
    upload_to_cloud "$DB_BACKUP"
    upload_to_cloud "$APP_BACKUP"
    [ ! -z "$REDIS_BACKUP" ] && upload_to_cloud "$REDIS_BACKUP"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Calculate backup sizes
    DB_SIZE=$(du -h "$DB_BACKUP" | cut -f1)
    APP_SIZE=$(du -h "$APP_BACKUP" | cut -f1)
    
    BACKUP_MESSAGE="✅ Backup completed successfully!
    
    Database backup: $DB_SIZE
    Application backup: $APP_SIZE
    Redis backup: $([ ! -z "$REDIS_BACKUP" ] && du -h "$REDIS_BACKUP" | cut -f1 || echo "Skipped")
    
    Files saved to: $BACKUP_DIR
    Cloud storage: $CLOUD_STORAGE"
    
    log "$BACKUP_MESSAGE"
    send_notification "Success" "$BACKUP_MESSAGE"
}

# Error handling
trap 'error "Backup failed with exit code $?"' ERR

# Run main function
main "$@"

log "🎉 Backup system completed successfully!"

# Crontab example (add with: crontab -e)
cat << 'EOF'

# Add this to crontab for automated daily backups:
# 0 2 * * * /path/to/free-backup-system.sh >> /var/log/backup.log 2>&1

# Weekly full backup:
# 0 2 * * 0 /path/to/free-backup-system.sh --full >> /var/log/backup.log 2>&1

EOF
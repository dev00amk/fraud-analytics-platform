#!/bin/bash
set -euo pipefail

# Blue-Green Deployment Script for Fraud Analytics Platform
# Implements zero-downtime deployment with automatic rollback

NAMESPACE="production"
APP_NAME="fraud-analytics-api"
HEALTH_CHECK_URL="https://fraudanalytics.dev/health/"
TIMEOUT=300
ROLLBACK_ON_FAILURE=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required environment variables
check_env_vars() {
    local required_vars=("IMAGE_TAG" "DATABASE_URL" "REDIS_URL" "SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
}

# Get current deployment color
get_current_color() {
    local current_selector=$(kubectl get service ${APP_NAME}-service -n ${NAMESPACE} -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "")
    if [[ "$current_selector" == "blue" ]]; then
        echo "blue"
    elif [[ "$current_selector" == "green" ]]; then
        echo "green"
    else
        echo "blue"  # Default to blue if no color is set
    fi
}

# Get target deployment color
get_target_color() {
    local current_color=$1
    if [[ "$current_color" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Deploy to target environment
deploy_target() {
    local target_color=$1
    local deployment_name="${APP_NAME}-${target_color}"
    
    log "Deploying to ${target_color} environment..."
    
    # Create deployment manifest with color-specific labels
    envsubst < k8s/production/deployment.yaml | \
    sed "s/name: ${APP_NAME}/name: ${deployment_name}/g" | \
    sed "s/app: ${APP_NAME}/app: ${APP_NAME}\n        color: ${target_color}/g" | \
    kubectl apply -f -
    
    # Wait for deployment to be ready
    log "Waiting for ${target_color} deployment to be ready..."
    if ! kubectl rollout status deployment/${deployment_name} -n ${NAMESPACE} --timeout=${TIMEOUT}s; then
        error "Deployment to ${target_color} environment failed"
        return 1
    fi
    
    success "${target_color} deployment is ready"
    return 0
}

# Health check for target deployment
health_check() {
    local target_color=$1
    local deployment_name="${APP_NAME}-${target_color}"
    local max_attempts=30
    local attempt=1
    
    log "Performing health checks on ${target_color} environment..."
    
    # Get pod IPs for direct health checks
    local pod_ips=$(kubectl get pods -n ${NAMESPACE} -l app=${APP_NAME},color=${target_color} -o jsonpath='{.items[*].status.podIP}')
    
    for pod_ip in $pod_ips; do
        log "Health checking pod at ${pod_ip}..."
        
        while [[ $attempt -le $max_attempts ]]; do
            if kubectl exec -n ${NAMESPACE} deployment/${deployment_name} -- curl -f -s http://${pod_ip}:8000/health/ > /dev/null; then
                success "Pod ${pod_ip} is healthy"
                break
            else
                warning "Health check attempt ${attempt}/${max_attempts} failed for pod ${pod_ip}"
                sleep 10
                ((attempt++))
            fi
        done
        
        if [[ $attempt -gt $max_attempts ]]; then
            error "Health check failed for pod ${pod_ip} after ${max_attempts} attempts"
            return 1
        fi
        
        attempt=1
    done
    
    # Additional application-specific health checks
    log "Running application-specific health checks..."
    
    # Check database connectivity
    if ! kubectl exec -n ${NAMESPACE} deployment/${deployment_name} -- python manage.py check --database default; then
        error "Database connectivity check failed"
        return 1
    fi
    
    # Check Redis connectivity
    if ! kubectl exec -n ${NAMESPACE} deployment/${deployment_name} -- python -c "
import redis
import os
r = redis.from_url(os.environ['REDIS_URL'])
r.ping()
print('Redis connection successful')
"; then
        error "Redis connectivity check failed"
        return 1
    fi
    
    # Check ML models loading
    if ! kubectl exec -n ${NAMESPACE} deployment/${deployment_name} -- python -c "
from apps.fraud_detection.services import AdvancedFraudDetectionService
service = AdvancedFraudDetectionService()
print('ML models loaded successfully')
"; then
        warning "ML models check failed, but continuing deployment"
    fi
    
    success "All health checks passed for ${target_color} environment"
    return 0
}

# Switch traffic to target deployment
switch_traffic() {
    local target_color=$1
    
    log "Switching traffic to ${target_color} environment..."
    
    # Update service selector to point to target color
    kubectl patch service ${APP_NAME}-service -n ${NAMESPACE} -p '{"spec":{"selector":{"color":"'${target_color}'"}}}'
    
    # Wait a moment for the change to propagate
    sleep 10
    
    # Verify traffic switch
    local service_selector=$(kubectl get service ${APP_NAME}-service -n ${NAMESPACE} -o jsonpath='{.spec.selector.color}')
    if [[ "$service_selector" == "$target_color" ]]; then
        success "Traffic successfully switched to ${target_color} environment"
        return 0
    else
        error "Failed to switch traffic to ${target_color} environment"
        return 1
    fi
}

# Perform smoke tests on live environment
smoke_tests() {
    local target_color=$1
    
    log "Running smoke tests on live ${target_color} environment..."
    
    # Basic endpoint tests
    local endpoints=(
        "/health/"
        "/api/v1/auth/token/"
        "/docs/"
    )
    
    for endpoint in "${endpoints[@]}"; do
        local url="${HEALTH_CHECK_URL%/}${endpoint}"
        log "Testing endpoint: ${url}"
        
        local response_code=$(curl -s -o /dev/null -w "%{http_code}" "${url}" || echo "000")
        
        if [[ "$endpoint" == "/api/v1/auth/token/" ]]; then
            # Auth endpoint should return 400 for missing credentials, not 500
            if [[ "$response_code" == "400" ]]; then
                success "Auth endpoint is responding correctly"
            else
                error "Auth endpoint returned unexpected status: ${response_code}"
                return 1
            fi
        else
            # Other endpoints should return 200
            if [[ "$response_code" == "200" ]]; then
                success "Endpoint ${endpoint} is healthy"
            else
                error "Endpoint ${endpoint} returned status: ${response_code}"
                return 1
            fi
        fi
    done
    
    # Performance test
    log "Running performance test..."
    local avg_response_time=$(curl -s -w "%{time_total}" -o /dev/null "${HEALTH_CHECK_URL}")
    local max_response_time=2.0
    
    if (( $(echo "${avg_response_time} < ${max_response_time}" | bc -l) )); then
        success "Performance test passed (${avg_response_time}s < ${max_response_time}s)"
    else
        warning "Performance test failed (${avg_response_time}s >= ${max_response_time}s)"
        # Don't fail deployment for performance issues, just warn
    fi
    
    success "All smoke tests passed"
    return 0
}

# Cleanup old deployment
cleanup_old_deployment() {
    local old_color=$1
    local old_deployment_name="${APP_NAME}-${old_color}"
    
    log "Cleaning up old ${old_color} deployment..."
    
    # Scale down old deployment
    kubectl scale deployment ${old_deployment_name} -n ${NAMESPACE} --replicas=0
    
    # Wait for pods to terminate
    kubectl wait --for=delete pod -l app=${APP_NAME},color=${old_color} -n ${NAMESPACE} --timeout=120s
    
    # Delete old deployment
    kubectl delete deployment ${old_deployment_name} -n ${NAMESPACE} --ignore-not-found=true
    
    success "Old ${old_color} deployment cleaned up"
}

# Rollback to previous deployment
rollback() {
    local current_color=$1
    local previous_color=$2
    
    error "Rolling back to ${previous_color} environment..."
    
    # Switch traffic back
    kubectl patch service ${APP_NAME}-service -n ${NAMESPACE} -p '{"spec":{"selector":{"color":"'${previous_color}'"}}}'
    
    # Scale up previous deployment if it exists
    local previous_deployment="${APP_NAME}-${previous_color}"
    if kubectl get deployment ${previous_deployment} -n ${NAMESPACE} &>/dev/null; then
        kubectl scale deployment ${previous_deployment} -n ${NAMESPACE} --replicas=3
        kubectl rollout status deployment/${previous_deployment} -n ${NAMESPACE} --timeout=120s
    fi
    
    # Clean up failed deployment
    kubectl delete deployment ${APP_NAME}-${current_color} -n ${NAMESPACE} --ignore-not-found=true
    
    error "Rollback completed. Service is running on ${previous_color} environment."
}

# Send deployment notification
send_notification() {
    local status=$1
    local target_color=$2
    local message=""
    
    if [[ "$status" == "success" ]]; then
        message="🎉 Fraud Analytics Platform successfully deployed to production (${target_color} environment)"
    else
        message="❌ Fraud Analytics Platform deployment failed and was rolled back"
    fi
    
    # Send to Slack if webhook is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"${message}\"}" \
            "${SLACK_WEBHOOK_URL}" || true
    fi
    
    # Send to Discord if webhook is configured
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"content\":\"${message}\"}" \
            "${DISCORD_WEBHOOK_URL}" || true
    fi
    
    log "Notification sent: ${message}"
}

# Main deployment function
main() {
    log "Starting blue-green deployment for Fraud Analytics Platform"
    
    # Check prerequisites
    check_env_vars
    
    # Determine current and target colors
    local current_color=$(get_current_color)
    local target_color=$(get_target_color "$current_color")
    
    log "Current environment: ${current_color}"
    log "Target environment: ${target_color}"
    log "Image tag: ${IMAGE_TAG}"
    
    # Deploy to target environment
    if ! deploy_target "$target_color"; then
        error "Deployment failed"
        send_notification "failure" "$target_color"
        exit 1
    fi
    
    # Health check target environment
    if ! health_check "$target_color"; then
        error "Health checks failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback "$target_color" "$current_color"
        fi
        send_notification "failure" "$target_color"
        exit 1
    fi
    
    # Switch traffic to target environment
    if ! switch_traffic "$target_color"; then
        error "Traffic switch failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback "$target_color" "$current_color"
        fi
        send_notification "failure" "$target_color"
        exit 1
    fi
    
    # Run smoke tests on live environment
    if ! smoke_tests "$target_color"; then
        error "Smoke tests failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback "$target_color" "$current_color"
        fi
        send_notification "failure" "$target_color"
        exit 1
    fi
    
    # Clean up old deployment
    cleanup_old_deployment "$current_color"
    
    # Send success notification
    send_notification "success" "$target_color"
    
    success "Blue-green deployment completed successfully!"
    success "Service is now running on ${target_color} environment with image ${IMAGE_TAG}"
}

# Run main function
main "$@"
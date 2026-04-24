#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
#  Guardian ML — Unified Launcher
#  Usage:
#    ./run.sh backend       → Start FastAPI server
#    ./run.sh frontend      → Serve Mode A frontend (static server)
#    ./run.sh static        → Serve GitHub Pages version locally
#    ./run.sh test          → Run pytest suite
#    ./run.sh install       → Install Python dependencies
#    ./run.sh all           → Install + start backend
# ═══════════════════════════════════════════════════════

set -e

BACKEND_DIR="$(cd "$(dirname "$0")/backend" && pwd)"
FRONTEND_DIR="$(cd "$(dirname "$0")/frontend" && pwd)"
STATIC_DIR="$(cd "$(dirname "$0")/github-pages" && pwd)"
TESTS_DIR="$(cd "$(dirname "$0")/tests" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

banner() {
  echo -e "${BLUE}"
  echo "  ╔══════════════════════════════════════╗"
  echo "  ║       GUARDIAN ML v1.0.0             ║"
  echo "  ║  Risk & Decision Support Platform    ║"
  echo "  ╚══════════════════════════════════════╝"
  echo -e "${NC}"
}

cmd_install() {
  echo -e "${YELLOW}→ Installing Python dependencies...${NC}"
  cd "$BACKEND_DIR"
  if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ Python 3 not found. Install Python 3.9+${NC}"
    exit 1
  fi
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  echo -e "${GREEN}✓ Dependencies installed.${NC}"
}

cmd_backend() {
  echo -e "${YELLOW}→ Starting FastAPI backend...${NC}"
  echo -e "  API: ${BLUE}http://localhost:8000${NC}"
  echo -e "  Docs: ${BLUE}http://localhost:8000/docs${NC}"
  cd "$BACKEND_DIR"
  python3 main.py
}

cmd_frontend() {
  echo -e "${YELLOW}→ Serving Mode A frontend on port 3000...${NC}"
  echo -e "  UI: ${BLUE}http://localhost:3000${NC}"
  echo -e "  ${YELLOW}⚠  Ensure backend is running on port 8000${NC}"
  cd "$FRONTEND_DIR"
  if command -v python3 &>/dev/null; then
    python3 -m http.server 3000
  elif command -v npx &>/dev/null; then
    npx serve . -p 3000
  else
    echo -e "${RED}✗ No static server available. Install Python 3 or Node.js${NC}"
    exit 1
  fi
}

cmd_static() {
  echo -e "${YELLOW}→ Serving GitHub Pages (Mode B) on port 4000...${NC}"
  echo -e "  URL: ${BLUE}http://localhost:4000${NC}"
  cd "$STATIC_DIR"
  if command -v python3 &>/dev/null; then
    python3 -m http.server 4000
  elif command -v npx &>/dev/null; then
    npx serve . -p 4000
  else
    echo -e "${RED}✗ No static server available.${NC}"
    exit 1
  fi
}

cmd_test() {
  echo -e "${YELLOW}→ Running test suite...${NC}"
  cd "$(dirname "$0")"
  python3 -m pytest tests/test_api.py -v --tb=short
}

cmd_all() {
  cmd_install
  cmd_backend
}

banner

case "${1:-help}" in
  install)  cmd_install  ;;
  backend)  cmd_backend  ;;
  frontend) cmd_frontend ;;
  static)   cmd_static   ;;
  test)     cmd_test     ;;
  all)      cmd_all      ;;
  *)
    echo -e "Usage: ${GREEN}./run.sh${NC} [command]"
    echo ""
    echo "Commands:"
    echo -e "  ${GREEN}install${NC}   Install Python dependencies"
    echo -e "  ${GREEN}backend${NC}   Start FastAPI ML server"
    echo -e "  ${GREEN}frontend${NC}  Serve Mode A UI (localhost:3000)"
    echo -e "  ${GREEN}static${NC}    Serve GitHub Pages UI (localhost:4000)"
    echo -e "  ${GREEN}test${NC}      Run pytest test suite"
    echo -e "  ${GREEN}all${NC}       Install + start backend"
    ;;
esac

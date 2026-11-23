#!/bin/bash
#
# ==============================================================================
#
#                              JETSON TOOLBOX
#
#                   Helper menu for the Jetson Orin Nano
#
# ==============================================================================
#
#
#   HOW TO RUN
#   ----------
#
#     First time only (makes the file runnable):
#
#         chmod +x ~/Desktop/jetson_tool.sh
#
#     Every time after that:
#
#         ~/Desktop/jetson_tool.sh
#
#     On a brand-new board, pick option 2 once, then reboot.
#
#
#   WIRING  (CP2102 module  ->  Jetson 40-pin header)
#   -------------------------------------------------
#
#         CP2102 TXD   ->   Pin 10  (RX)    5th pin in the even row
#         CP2102 RXD   ->   Pin 8   (TX)    4th pin in the even row
#         CP2102 GND   ->   Pin 6   (GND)   3rd pin in the even row
#
#     Do NOT connect the module's 3.3V / 5V pin.
#     Do NOT touch pins 2 and 4 (5V).
#
#
#   MENU OPTIONS
#   ------------
#
#     1)  Fix clock            Board loses time when unplugged
#     2)  First-time setup     Installs picocom + pyserial, fixes permissions
#     3)  List serial ports    Shows ttyTHS / ttyUSB / ttyACM devices
#     4)  Change port / baud   Default is /dev/ttyTHS1 at 115200
#     5)  Open serial terminal Exit with Ctrl+A, then Ctrl+X
#     6)  UART loopback test   Jumper pin 8 to pin 10 first
#     7)  Run a Python script  e.g. ~/Desktop/aim.py
#     8)  System info          Temp, memory, disk, IP address
#
#
#   ON THE MAC SIDE
#   ---------------
#
#         ls /dev/cu.*                              (find the usbserial name)
#         screen /dev/cu.usbserial-XXXX 115200      (use your real name)
#
#     Exit screen with Ctrl+A, then K, then Y.
#
#
#   ON THE WINDOWS SIDE
#   -------------------
#
#     1.  Open Device Manager  ->  Ports (COM & LPT)
#         Find "Silicon Labs CP210x" and note the COM number (e.g. COM5).
#
#         Not listed?  Install the CP210x VCP driver from Silicon Labs,
#         then unplug and replug the module.
#
#     2.  Open PuTTY (free download from putty.org):
#
#         Connection type:   Serial
#         Serial line:       COM5          (use your real number)
#         Speed:             115200
#
#         Click Open.  Close the window to exit.
#
#     No PuTTY?  In PowerShell you can find the port with:
#
#         [System.IO.Ports.SerialPort]::GetPortNames()
#
#
#   ADDING YOUR OWN OPTIONS
#   -----------------------
#
#     Look for the two "ADD YOUR OWN" lines near the bottom:
#
#       1. Add a menu line:   echo " 9) Start auto-aim"
#       2. Add the action:    9) sudo python3 ~/Desktop/aim.py ;;
#
#
# ==============================================================================


PORT="/dev/ttyTHS1"   # 40-pin header UART (pins 8 TX / 10 RX)
BAUD="115200"

pause() { read -rp "Press Enter to continue..."; }

fix_clock() {
  echo "Current time: $(date)"
  echo "1) Sync from internet   2) Type it in manually"
  read -rp "> " c
  if [ "$c" = "1" ]; then
    sudo timedatectl set-ntp true && sleep 3 && echo "Now: $(date)"
  else
    read -rp "Enter date/time (e.g. 2026-10-27 14:30:00): " t
    sudo timedatectl set-ntp false
    sudo date -s "$t"
  fi
}

setup_once() {
  echo "Installing tools and fixing serial permissions..."
  sudo apt update
  sudo apt install -y picocom python3-serial
  sudo usermod -aG dialout "$USER"
  echo "Done. Reboot once for the permission change to take effect."
}

list_ports() {
  echo "Serial ports on this board:"
  ls /dev/ttyTHS* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "(none found)"
  echo "Currently using: $PORT @ $BAUD"
}

change_port() {
  list_ports
  read -rp "New port (Enter to keep $PORT): " p; [ -n "$p" ] && PORT="$p"
  read -rp "New baud (Enter to keep $BAUD): " b; [ -n "$b" ] && BAUD="$b"
}

open_terminal() {
  echo "Opening $PORT @ $BAUD  (exit: Ctrl+A then Ctrl+X)"
  sudo picocom -b "$BAUD" --echo "$PORT"
}

loopback_test() {
  echo "Put a jumper between pin 8 and pin 10, then press Enter."
  pause
  sudo python3 - "$PORT" "$BAUD" <<'EOF'
import serial, sys
s = serial.Serial(sys.argv[1], int(sys.argv[2]), timeout=1)
s.reset_input_buffer()
s.write(b'loopback-test')
r = s.read(13)
print("PASS - UART works" if r == b'loopback-test' else f"FAIL - got {r!r}")
EOF
}

run_python() {
  read -rp "Path to Python script (e.g. ~/Desktop/aim.py): " f
  f="${f/#\~/$HOME}"
  [ -f "$f" ] && sudo python3 "$f" || echo "File not found: $f"
}

system_info() {
  echo "Uptime: $(uptime -p)"
  echo "CPU temp: $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )) C"
  free -h | head -2
  df -h / | tail -1
  ip -4 addr show | grep inet | grep -v 127.0.0.1
}

while true; do
  clear
  echo "========= JETSON TOOLBOX ========="
  echo " Port: $PORT  Baud: $BAUD"
  echo "----------------------------------"
  echo " 1) Fix clock"
  echo " 2) First-time setup (install tools)"
  echo " 3) List serial ports"
  echo " 4) Change port / baud"
  echo " 5) Open serial terminal"
  echo " 6) UART loopback test"
  echo " 7) Run a Python script"
  echo " 8) System info (temp, memory, IP)"
  # ===== ADD YOUR OWN: add a line here, e.g. " 9) Start auto-aim" =====
  echo " q) Quit"
  read -rp "> " choice
  case "$choice" in
    1) fix_clock ;;
    2) setup_once ;;
    3) list_ports ;;
    4) change_port ;;
    5) open_terminal ;;
    6) loopback_test ;;
    7) run_python ;;
    8) system_info ;;
    # ===== ADD YOUR OWN: e.g.  9) sudo python3 ~/Desktop/aim.py ;; =====
    q) exit 0 ;;
    *) echo "Unknown option" ;;
  esac
  pause
done

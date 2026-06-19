#!/usr/bin/python3
import serial
import sys
import termios
import tty
import atexit
import os
import time

# ========================= CONFIG =========================
SERIAL_PORT = "/dev/ttyAMA0"
BAUDRATE = 9600
TIMEOUT = 0.5
STEP_SIZE = 3

PAN_MIN = -90
PAN_MAX = 90

# === Servo values sent to hardware (DO NOT CHANGE THESE) ===
SERVO_TILT_UPRIGHT = 30      # servo command when gimbal is straight up
SERVO_TILT_MAX_SAFE = 75     # Adjusted so display stops at 75 degrees

# === Displayed values ===
DISPLAY_UPRIGHT = 0          # shown when straight up
DISPLAY_MAX_TILT = 65        # now set to 75 as requested

DEFAULT_PAN = 0
DEFAULT_SERVO_TILT = 65      # safe starting servo value
# =========================================================

def servo_to_display(servo_angle):
    """Convert internal servo value to user-friendly display angle"""
    if servo_angle <= SERVO_TILT_UPRIGHT:
        return DISPLAY_UPRIGHT
    if servo_angle >= SERVO_TILT_MAX_SAFE:
        return DISPLAY_MAX_TILT
    
    # Linear mapping: servo 30→0, servo 85→75
    ratio = (servo_angle - SERVO_TILT_UPRIGHT) / (SERVO_TILT_MAX_SAFE - SERVO_TILT_UPRIGHT)
    display_angle = DISPLAY_UPRIGHT + ratio * (DISPLAY_MAX_TILT - DISPLAY_UPRIGHT)
    return round(display_angle)


def send_command(ser, channel, angle):
    angle = int(angle)
    
    if channel == 'A':   # Pan
        angle = max(PAN_MIN, min(PAN_MAX, angle))
        servo_angle = angle + 90
        display_angle = angle
    else:                # Tilt
        servo_angle = max(SERVO_TILT_UPRIGHT, min(SERVO_TILT_MAX_SAFE, angle))
        display_angle = servo_to_display(servo_angle)
    
    cmd = f"${channel}{servo_angle:03d}#"
    ser.write(cmd.encode('utf-8'))
    print(f"Sent: {cmd} | Channel {channel}: {display_angle} degrees (servo: {servo_angle} degrees)")


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1).decode('utf-8', errors='ignore')
        if ch == '\x1b':
            os.read(fd, 2)
            return ''
        return ch.upper()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def return_to_default(ser):
    if not ser or not ser.is_open:
        return
    print("\nReturning gimbal to default position...")
    send_command(ser, 'A', DEFAULT_PAN)
    time.sleep(1.0)
    send_command(ser, 'B', DEFAULT_SERVO_TILT)
    time.sleep(0.6)
    print(f"Default position restored -> Pan: {DEFAULT_PAN} degrees | Tilt: {servo_to_display(DEFAULT_SERVO_TILT)} degrees")


def main():
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, 8, 'N', 1, timeout=TIMEOUT)
        print("Serial port opened successfully")
    except Exception as e:
        print(f"Failed to open {SERIAL_PORT}: {e}")
        sys.exit(1)

    atexit.register(return_to_default, ser)

    print("\n" + "=" * 80)
    print("Gimbal Control")
    print("=" * 80)
    print("W -> Tilt UP     (more upright / perpendicular)")
    print("S -> Tilt DOWN   (toward safe tilt position)")
    print("A -> Pan Left    D -> Pan Right")
    print("Q -> Quit")
    print("=" * 80)
    print(f"Tilt range : {DISPLAY_UPRIGHT} degrees (straight up) -> {DISPLAY_MAX_TILT} degrees (max safe tilt)")
    print(f"Pan  range : {PAN_MIN} degrees <- Center(0) -> {PAN_MAX} degrees\n")

    pan_angle = DEFAULT_PAN
    servo_tilt_angle = DEFAULT_SERVO_TILT

    send_command(ser, 'A', pan_angle)
    send_command(ser, 'B', servo_tilt_angle)

    print(f"Started at default position -> Pan: {pan_angle} degrees | Tilt: {servo_to_display(servo_tilt_angle)} degrees\n")

    try:
        while True:
            key = get_key()
            moved = False

            if key == 'Q':
                print("\nExiting program...")
                break

            elif key == 'A':        # Pan Left
                new_angle = pan_angle + STEP_SIZE
                if new_angle <= PAN_MAX:
                    pan_angle = new_angle
                    send_command(ser, 'A', pan_angle)
                    moved = True
                else:
                    print(f"Pan left limit reached ({PAN_MAX} degrees)")

            elif key == 'D':        # Pan Right
                new_angle = pan_angle - STEP_SIZE
                if new_angle >= PAN_MIN:
                    pan_angle = new_angle
                    send_command(ser, 'A', pan_angle)
                    moved = True
                else:
                    print(f"Pan right limit reached ({PAN_MIN} degrees)")

            elif key == 'W':        # Tilt UP
                new_servo = servo_tilt_angle - STEP_SIZE
                if new_servo >= SERVO_TILT_UPRIGHT:
                    servo_tilt_angle = new_servo
                    send_command(ser, 'B', servo_tilt_angle)
                    moved = True
                else:
                    print(f"Tilt UP limit reached ({DISPLAY_UPRIGHT} degrees - straight up)")

            elif key == 'S':        # Tilt DOWN
                new_servo = servo_tilt_angle + STEP_SIZE
                if new_servo <= SERVO_TILT_MAX_SAFE:
                    servo_tilt_angle = new_servo
                    send_command(ser, 'B', servo_tilt_angle)
                    moved = True
                else:
                    print(f"Tilt DOWN limit reached ({DISPLAY_MAX_TILT} degrees - max safe tilt)")

            elif key == '':
                continue
            else:
                print(f"Unknown key '{key}' -> Use W/A/S/D or Q")

            if moved:
                display_tilt = servo_to_display(servo_tilt_angle)
                print(f"Current position -> Pan: {pan_angle} degrees | Tilt: {display_tilt} degrees")

    except KeyboardInterrupt:
        print("\nProgram terminated by user (Ctrl+C)")
    finally:
        if ser and ser.is_open:
            return_to_default(ser)
            ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()

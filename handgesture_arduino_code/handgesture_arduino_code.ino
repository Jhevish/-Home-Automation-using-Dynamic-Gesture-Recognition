const int ledPin = 9;
const int motorPin = 10;

int light_intensity = 0;
int motor_speed = 0;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(motorPin, OUTPUT);
  analogWrite(ledPin, light_intensity);
  analogWrite(motorPin, motor_speed);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
   
    if (msg.startsWith("L")) {
      int value = msg.substring(1).toInt();
      light_intensity = map(value, 0, 4, 0, 255);
      analogWrite(ledPin, light_intensity);
    } else if (msg.startsWith("F")) {
      int value = msg.substring(1).toInt();
      motor_speed = map(value, 0, 4, 0, 255);
      analogWrite(motorPin, motor_speed);
    }
  }
}

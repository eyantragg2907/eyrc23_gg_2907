#include <WiFi.h>

const char *ssid = "pjrWifi";          // Enter your wifi hotspot ssid
const char *password = "SimplePass01"; // Enter your wifi hotspot password
const uint16_t port = 8002;
const char *host = "192.168.229.92";

const int IR1 = 5; // IR sensors pins
const int IR2 = 18;
const int IR3 = 32;
const int IR4 = 33;
const int IR5 = 25;

const int motor1f = 12; // motor LEFT
const int motor1r = 14;

const int motor2f = 13; // motor RIGHT
const int motor2r = 27;

const int led_red = 2; // misc
const int led_green = 15;
const int buzzer = 23;

int speed1 = 255; // motor speeds
int speed2 = 255;

int turntime = 1000; // in milliseconds

int input1, input2, input3, input4, input5; // inputs of IR sensors, returns LOW when black line

bool mode = 0; // 0 is bang bang, 1 is the middle line

// String path = "nnnrlrrnrnln"; // n means do nothing, l = left, r = right
String path = "";

char incomingPacket[80];
WiFiClient client;

String msg = "0";
int counter = 0;

int i = 0; // flags
int move = false;
bool atNode = true;
void forward()
{
  Serial.println("MOVING FORWARD");
  if (mode == 0)
  { // bang bang
    if (!input1)
    { // left IR hits black
      analogWrite(motor1f, speed1);
    }
    else if (!input5)
    { // right IR hits black
      analogWrite(motor2f, speed2);
    }
    else
    {
      analogWrite(motor1f, speed1);
      analogWrite(motor2f, speed2);
    }
  }
  else
  { // middle black line
    if (!input3)
    { // following the line
      analogWrite(motor1f, speed1);
      analogWrite(motor2f, speed2);
    }
    if (!input4)
    { // right IR touches black
      analogWrite(motor1f, speed1);
    }
    if (!input2)
    { // left IR touches black
      analogWrite(motor2f, speed2);
    }
  }
}
void stop()
{
  analogWrite(motor1f, 0);
  analogWrite(motor2f, 0);
  analogWrite(motor1r, 0);
  analogWrite(motor2r, 0);
}
void goToNextNode()
{
  atNode = false;
  if (!(input2 || input3 || input4))
  { // go ahead until middle IRs see black
    Serial.println("black forward march");
    forward();
  }
  else
  {
    stop();
    Serial.println("NODE");
    atNode = true;
    digitalWrite(buzzer, LOW);
    delay(1000);
    digitalWrite(buzzer, HIGH);
  }
}

void turn(char x)
{
  stop();
  if (x == 'l')
  {
    analogWrite(motor2f, speed2);
    analogWrite(motor1r, speed1);
    delay(turntime);
  }
  else
  {
    analogWrite(motor1f, speed1);
    analogWrite(motor2r, speed2);
  }
  stop();
}
void setup()
{

  // set  modes
  pinMode(IR1, INPUT);
  pinMode(IR2, INPUT);
  pinMode(IR3, INPUT);
  pinMode(IR4, INPUT);
  pinMode(IR5, INPUT);

  pinMode(motor1f, OUTPUT);
  pinMode(motor1r, OUTPUT);
  pinMode(motor2f, OUTPUT);
  pinMode(motor2r, OUTPUT);

  pinMode(led_red, OUTPUT);
  pinMode(buzzer, OUTPUT);

  digitalWrite(buzzer, HIGH);

  Serial.begin(115200);

  // setting up wifi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.println("...");
  }

  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());

  digitalWrite(led_red, HIGH); // rest condition
  digitalWrite(led_green, LOW);

  do
  {
    Serial.println("Connection to host failed");
    // digitalWrite(buzzer, LOW);
    digitalWrite(led_red, HIGH);
    delay(200);
    // digitalWrite(buzzer, HIGH);
  } while (!client.connect(host, port));

  digitalWrite(led_red, LOW);
  client.print("Obese American ate 69 giant ramen bowl but still is lighter than your mom"); // Send an acknowledgement to host(laptop)
  msg = client.readStringUntil('\n');                                                        // Read the message through the socket until new line char(\n)
  path = msg;
  move = true;
  Serial.println(path);
}

void loop()
{
  input1 = digitalRead(IR1); // read IR sensors
  input2 = digitalRead(IR2);
  input3 = digitalRead(IR3);
  input4 = digitalRead(IR4);
  input5 = digitalRead(IR5);

  if (!(input2 && input3 && input4))
  { // switch to correct mode
    mode = 0;
  }
  else
  {
    mode = 1;
  }
  Serial.print("mode ");
  Serial.println(mode);

  if (move)
  { // start moving
    Serial.println("MOVING...");
    digitalWrite(led_red, LOW);
    digitalWrite(led_green, HIGH);
    if (atNode)
    {
      Serial.print("At a node... ");
      char command = path[i++];
      Serial.println(command);
      if (command == 'l')
      {
        turn(command);
      }
      if (command == 'r')
      {
        turn(command);
      }
      Serial.println("Moving ahead!");
      analogWrite(motor1f, speed1);
      analogWrite(motor2f, speed2); // pushing the robot out of the node
    }
    Serial.println("Go to next");
    goToNextNode();

    if (i == path.length())
    {
      Serial.println("path complete");
      move = false; // break out of the loop
      stop();
      digitalWrite(led_green, LOW);
      digitalWrite(led_red, HIGH);
      digitalWrite(buzzer, LOW);
      delay(5000);
      digitalWrite(led_red, LOW);
      digitalWrite(buzzer, HIGH);
    }
  }
}

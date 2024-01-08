/* A set of quick-fire tests */
#define WIFI 0

const char *ssid = "pjrWifi";
const char *password = "SimplePass01";
const uint16_t port = 8002;
const char *host = "192.168.128.92";

const int IR1 = 5; // IR sensors pins
const int IR2 = 18;
const int IR3 = 32;
const int IR4 = 33;
const int IR5 = 25;

const int motor1f = 12; // motor LEFT forward
const int motor1r = 14; // motor LEFT reverse

const int motor2f = 13; // motor RIGHT forward
const int motor2r = 27; // motor RIGHT reverse

const int led_red = 2; // misc
const int led_green = 15;
const int buzzer = 23;

#include <WiFi.h>

WiFiClient client;

void connectToWifi()
{
    // setting up wifi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(WIFI_TRY_DELAY);
        Serial.println("...");
    }

    Serial.print("WiFi connected with IP: ");
    Serial.println(WiFi.localIP());

    digitalWrite(led_red, HIGH); // rest condition
    digitalWrite(led_green, LOW);

    do
    {
        Serial.println("Connection to host failed");
        digitalWrite(led_red, HIGH);
        delay(CONNECTION_PING_DELAY);
    } while (!client.connect(host, port));

    client.print("ACK_REQ_FROM_ROBOT"); // Send an acknowledgement to host(laptop)
}

void setup() {
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

    digitalWrite(buzzer, HIGH); // buzzer OFF is on HIGH

    Serial.begin(115200);

    // wifi connection 
    if (WIFI) {
        connectToWifi();
    }
}

void readIRs() {
    int in1, in2, in3, in4, in5;

    in1 = digitalRead(IR1);
    in2 = digitalRead(IR2);
    in3 = digitalRead(IR3);
    in4 = digitalRead(IR4);
    in5 = digitalRead(IR5);
    
    char s[12];
    snprintf(s, 12, "%d %d %d %d %d\n", in1, in2, in3, in4, in5);

    Serial.println(s);
    client.print(s);
}

void motorTest() {
    analogWrite(motor1f, 255);
    digitalWrite(motor1f, LOW);
    analogWrite(motor1r, 255);
    digitalWrite(motor1r, LOW);

    analogWrite(motor2f, 255);
    digitalWrite(motor2f, LOW);
    analogWrite(motor2r, 255);
    digitalWrite(motor2r, LOW);
}

void buzzerTest() {
    digitalWrite(buzzer, LOW);
    delay(1000);
    digitalWrite(buzzer, HIGH);
}

void ledsTest() {
    digitalWrite(led_red, HIGH);
    digitalWrite(led_green, HIGH);
    delay(1000);
    digitalWrite(led_red, LOW);
    digitalWrite(led_green, LOW);
}

void teleop() {
    String move = client.readStringUntil("\n");
    char to_move = move[0];

    if (to_move == 'F') {
        analogWrite(motor1f, 150);
        analogWrite(motor2f, 150);
    } else if (to_move == 'L') {
        analogWrite(motor1f, 150);
        digitalWrite(motor1r, LOW);
        digitalWrite(motor2f, LOW);
        analogWrite(motor2r, 150);
    } else if (to_move == 'R') {
        digitalWrite(motor1f, LOW);
        analogWrite(motor1r, 150);
        analogWrite(motor2f, 150);
        digitalWrite(motor2r, LOW);
    } else if (to_move == 'B') {
        analogWrite(motor1r, 150);
        analogWrite(motor2r, 150);
    }
}

void loop() {
    // buzzerTest();
    // ledsTest();
    // readIRs();
    teleop();
    readIRs();
}
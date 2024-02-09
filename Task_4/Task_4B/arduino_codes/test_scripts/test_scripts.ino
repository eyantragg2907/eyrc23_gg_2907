/* A set of quick-fire tests */
#define WIFI 0
#define MOVE_SPEED 180
#define MOTORTEST_SPEED 255

// const char *ssid = "pjrWifi";
// const char *password = "SimplePass01";
// const uint16_t port = 8002;
// const char *host = "192.168.187.144";

const char *ssid = "brainerd";
const char *password = "internetaccess";
const uint16_t port = 8002;
const char *host = "192.168.56.1";

// const int IR1 = 18; // IR sensors pins ACTUALLY IR2
// const int IR2 = 25; // ACTUALLY IR5
// const int IR3 = 32;
// const int IR4 = 5; // ACTUALLY IR2
// const int IR5 = 33; // ACTUALLY IR4

const int IR1 = 5; // IR sensors pins
const int IR2 = 25;
const int IR3 = 32;
const int IR4 = 33;
const int IR5 = 18;

// looking from the back

const int motor1f = 27; // motor LEFT forward
const int motor1r = 13; // moto r LEFT reverse
const int motor2f = 12; // motor RIGHT forward
const int motor2r = 22; // motor RIGHT reverse, used to be 14

const int led_red = 2; // misc
const int led_green = 15;
const int buzzer = 23;

// #include <WiFi.h>

// WiFiClient client;

/*
void connectToWifi()
{
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
    digitalWrite(led_green, 0);

    do
    {
        Serial.println("Connection to host failed");
        digitalWrite(led_red, HIGH);
        delay(200);
    } while (!client.connect(host, port));

    client.print("ACK_REQ_FROM_ROBOT"); // Send an acknowledgement to host(laptop)
}
*/

void setup()
{
    Serial.begin(115200);

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
}

void readIRs()
{
    int in1, in2, in3, in4, in5;

    in1 = digitalRead(IR1);
    in2 = digitalRead(IR2);
    in3 = digitalRead(IR3);
    in4 = digitalRead(IR4);
    in5 = digitalRead(IR5);

    char s[12];
    snprintf(s, 12, "%d %d %d %d %d", in1, in2, in3, in4, in5);

    Serial.println(s);
    // client.print(s);
}

bool test_ran = false;
void motorTest()
{
    if (!test_ran)
    {
        Serial.println("STARTING test");
        Serial.println("M1F");
        analogWrite(motor1f, MOTORTEST_SPEED);
        delay(5000);
        analogWrite(motor1f, 0);
        Serial.println("M1R");
        analogWrite(motor1r, MOTORTEST_SPEED);
        delay(5000);
        analogWrite(motor1r, 0);

        Serial.println("M2F");
        analogWrite(motor2f, MOTORTEST_SPEED);
        delay(5000);
        analogWrite(motor2f, 0);
        Serial.println("M2R");
        analogWrite(motor2r, MOTORTEST_SPEED);
        delay(5000);
        analogWrite(motor2r, 0);
        Serial.println("DONE test");
    }
    test_ran = true;
}

void buzzerTest()
{
    digitalWrite(buzzer, 0);
    delay(1000);
    digitalWrite(buzzer, HIGH);
}

void ledsTest()
{
    digitalWrite(led_red, HIGH);
    // digitalWrite(led_green, HIGH);
    delay(1000);
    digitalWrite(led_red, 0);
    // digitalWrite(led_green, 0);
}

char to_move = '0';

/*
void teleop()
{
    if (!WIFI)
    {
        Serial.println("Cannot teleop without WiFi");
        return;
    }
    String move = client.readStringUntil('\n');
    if (move.length() != 0)
    {
        to_move = move[0];
    }

    // analogWrite(motor1r, 0);
    // analogWrite(motor2r, 0);
    // analogWrite(motor2f, 0);
    // analogWrite(motor1f, 0);

    if (to_move == 'F')
    {
        analogWrite(motor1r, 0);
        analogWrite(motor2r, 0);
        analogWrite(motor1f, MOVE_SPEED);
        analogWrite(motor2f, MOVE_SPEED);
    }
    else if (to_move == 'L')
    {
        analogWrite(motor1r, 0);
        analogWrite(motor2f, 0);
        analogWrite(motor1f, MOVE_SPEED);
        analogWrite(motor2r, MOVE_SPEED);
    }
    else if (to_move == 'R')
    {
        analogWrite(motor1f, 0);
        analogWrite(motor2r, 0);
        analogWrite(motor1r, MOVE_SPEED);
        analogWrite(motor2f, MOVE_SPEED);
    }
    else if (to_move == 'B')
    {
        analogWrite(motor1f, 0);
        analogWrite(motor2f, 0);
        analogWrite(motor1r, MOVE_SPEED);
        analogWrite(motor2r, MOVE_SPEED);
    }
    else if (to_move == 'S')
    {
        analogWrite(motor1r, 0);
        analogWrite(motor1f, 0);
        analogWrite(motor2r, 0);
        analogWrite(motor2f, 0);
    }
}
*/

void loop()
{
    buzzerTest();
    // ledsTest();

    // delay(5000);
    //readIRs();
    // teleop();
    // readIRs();

    // delay(200);
    // motorTest();

    // int in3 = digitalRead(IR3);
    // digitalWrite(led_red, in3);
}

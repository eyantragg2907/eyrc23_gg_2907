/*

THIS IS THE SKETCH WITH DEBUG COMMANDS AND STUFF FOR EFFICIENT PERFORMANCE

*/

#include <WiFi.h>

#define SPEED_LEFT 255  // default motor LEFT speed
#define SPEED_RIGHT 255 // default motor RIGHT speed

#define BANGBANG_TURNSPEED 150
#define MIDDLE_TURNSPEED 150

#define ROTATE_SPEED 120

#define ROT_COMPLETE_DELAY 100
#define LEAVE_BLACK_DELAY 150
#define NODE_LEAVE_DELAY 500

#define CENTER_CORRECT_DELAY 600

#define CONNECTION_PING_DELAY 200
#define WIFI_TRY_DELAY 500

#define EVERY_NODE_DELAY 1000
#define END_DELAY 5000

#define D90_TURNTIME 700

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

/* Status Variables */

int input1, input2, input3, input4, input5; // inputs of IR sensors, returns HIGH when black line

bool node = true;

WiFiClient client;

String msg = "";

int command_counter = 0; // flags
int operation = 0;       // 0 for forward, 1 for check next command, 2 for rotating left, 3 for rotating right, 4 for leaving the node, 5 terminating, 6 found node now what we do
int rotflag = 0;

// stops all motors
void stop()
{
    client.print("ALL STOP\n");

    analogWrite(motor1f, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1r, 0);
    analogWrite(motor2r, 0);
}

void printIRs()
{
    char s[12];
    snprintf(s, 12, "%d %d %d %d %d\n", input1, input2, input3, input4, input5);

    client.print(s);
}

void printMetaSerial()
{
    Serial.print(input1);
    Serial.print(" ");
    Serial.print(input2);
    Serial.print(" ");
    Serial.print(input3);
    Serial.print(" ");
    Serial.print(input4);
    Serial.print(" ");
    Serial.println(input5);
    Serial.println(operation);
}

// just moving forward (two controllers)
void moveForwardLogic()
{
    printIRs();
    if (input2 == 0 && input3 == 0 && input4 == 0) // bang bang controller
    {
        if (input1 == 1 && input5 == 0) // left line detected by left sensor
        {
            analogWrite(motor1f, BANGBANG_TURNSPEED);
            analogWrite(motor2f, 0);
        }
        else if (input5 == 1 && input1 == 0) // right line detected by right sensor
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, BANGBANG_TURNSPEED);
        }
        else
        {
            analogWrite(motor1f, SPEED_LEFT);
            analogWrite(motor2f, SPEED_RIGHT);
        }
    }
    else
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
        {
            analogWrite(motor1f, SPEED_LEFT);
            analogWrite(motor2f, SPEED_RIGHT);
        }
        else if (input2 == 1) // middle line detected by middle left sensor
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, MIDDLE_TURNSPEED);
        }
        else if (input4 == 1) // middle line detected by middle right sensor
        {
            analogWrite(motor1f, MIDDLE_TURNSPEED);
            analogWrite(motor2f, 0);
        }
    }
}

int moveForwardTillReachedNode()
{
    if (!(input2 == 1 && input3 == 1 && input4 == 1)) // if not at a node
    {
        moveForwardLogic();
        return 0;
    }
    else
    { // reached a node
        client.print("REACHED A NODE\n");
        printIRs();
        stop();
        return 1;
    }
}

void turn_left()
{
    analogWrite(motor1r, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1f, ROTATE_SPEED);
    analogWrite(motor2r, ROTATE_SPEED);
}

void turn_right()
{
    analogWrite(motor1r, ROTATE_SPEED);
    analogWrite(motor2f, ROTATE_SPEED);
    analogWrite(motor2r, 0);
    analogWrite(motor1f, 0);
}

int timeTurn(int dirn) {
    stop();
    if (node) {
        analogWrite(motor1f, SPEED_LEFT);
        analogWrite(motor2f, SPEED_RIGHT);
        client.print("ALIGNING CENTER OF ROTATION\n");
        delay(CENTER_CORRECT_DELAY);
        node = false;
    } else {
        if (dirn == 1) {
            turn_right();
        } else {
            turn_left();
        }
    }
    delay(D90_TURNTIME);
    stop();
    delay(ROT_COMPLETE_DELAY);
    node = true;
    return 1;
}

// dirn == 1 for RIGHT turns; else 0
int turn(int dirn)
{
    if (node)
    { // at a node
        // moved forward to align center of rotation
        analogWrite(motor1f, SPEED_LEFT);
        analogWrite(motor2f, SPEED_RIGHT);
        client.print("ALIGNING CENTER OF ROTATION\n");
        delay(CENTER_CORRECT_DELAY);
        node = false;
    } // Now we have left the node for sure!
    else
    {
        if (rotflag == 0) // rotate a little bit to leave the middle black line
        {
            if (dirn == 1)
            {
                turn_right();
            }
            else // left
            {
                turn_left();
            }
            delay(LEAVE_BLACK_DELAY);
            rotflag = 1;
            client.print("DETECT BLACK_LINE START\n");
            Serial.println("Rotated to leave the middle black line!");
        }
        else
        {
            if (dirn == 1) // rotate right
            {
                printIRs();
                if (input3 == 1) // reached the middle line again, we completed rotation
                {
                    client.print("BLACK_LINE DETECTED\n");
                    Serial.println("Rotation Completed");
                    rotflag = 0;
                    node = true;
                    stop();
                    delay(ROT_COMPLETE_DELAY);
                    return 1;
                }
                else
                {
                    turn_right();
                }
            }
        }
        else // rotate left
        {
            printIRs();
            if (input3 == 1)
            {
                client.print("BLACK_LINE DETECTED\n");
                Serial.println("Rotation Completed");
                rotflag = 0;
                node = true;
                stop();
                delay(ROT_COMPLETE_DELAY);
                return 1;
            }
            else
            {
                turn_left();
            }
        }
    }
    return 0;
}

String connectToWifiAndGetMessage()
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

    pathmsg = client.readStringUntil('\n'); // Read the message through the socket until new line char(\n)

    client.print(String("RECV: ") + pathmsg + String("\n"));

    return pathmsg;
}

/* ARDUINO FUNCTIONS */

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

    digitalWrite(buzzer, HIGH); // buzzer OFF is on HIGH

    Serial.begin(115200);

    msg = connectToWifiAndGetMessage();

    Serial.println(msg);

    Serial.println("Starting the run!");

    client.print("====-====\nSTART\n====-====\n");
    digitalWrite(led_red, LOW);
    digitalWrite(led_green, HIGH);
}

void loop()
{
    input1 = digitalRead(IR1);
    input2 = digitalRead(IR2);
    input3 = digitalRead(IR3);
    input4 = digitalRead(IR4);
    input5 = digitalRead(IR5);

    // printMetaSerial();

    if (operation == 0) // move forward
    {
        if (moveForwardTillReachedNode())
        {
            operation = 6;
        }
    }
    else if (operation == 1) // Next Command
    {
        client.print(String("index to read is ") + String(command_counter) + String("\n"));
        char command = msg[command_counter++];
        if (command == 'l')
        {
            client.print("-----> NEXT: left\n");
            Serial.println("Command to rotate left");
            operation = 2;
        }
        else if (command == 'r')
        {
            client.print("-----> NEXT: right\n");
            Serial.println("Command to rotate right");
            operation = 3;
        }
        else
        {
            client.print("-----> NEXT: forward\n");
            Serial.println("Command to move normally");
            operation = 4;
        }
    }
    else if (operation == 2) // rotate left
    {
        if (turn(0))
        {
            operation = 1;
        }
    }
    else if (operation == 3) // rotate right
    {
        if (turn(1))
        {
            operation = 1;
        }
    }
    else if (operation == 4) // leave the current node
    {
        if (input2 == 1 && input3 == 1 && input4 == 1)
        { // at a node
            analogWrite(motor1f, SPEED_LEFT);
            analogWrite(motor2f, SPEED_RIGHT);
            client.print("LEAVING NODE");
            delay(NODE_LEAVE_DELAY);
        }
        else if (command_counter == msg.length())
        {
            stop();
            Serial.println("path complete, starting the terminating sequence");
            operation = 5;
        }
        else
        {
            stop();
            client.print("LEFT Node, now moving forward");
            operation = 0;
        }
    }
    else if (operation == 5) // terminate
    {
        Serial.println("Going to the ending node!");
        client.print("END OF JOURNEY MOVEMENT START\n");

        moveForwardLogic();

        // TODO: implement history logic
        if (input3 == 0 && input2 == 0 && input4 == 0) // stop sign reached
        {
            client.print("REACHED NO LINE AREA\n");
            printIRs();
            Serial.println("Reached the ending node!");
            stop();
            digitalWrite(led_green, LOW);
            digitalWrite(led_red, HIGH);
            digitalWrite(buzzer, LOW);
            delay(END_DELAY);
            digitalWrite(led_red, LOW);
            digitalWrite(buzzer, HIGH);
            operation = -1;
        }
    }
    else if (operation == 6) // Node Found now what to do
    {
        digitalWrite(buzzer, LOW);
        delay(EVERY_NODE_DELAY);
        digitalWrite(buzzer, HIGH);
        operation = 1;
    }
    else
    {
        client.print("PATH COMPLETE\n");
        Serial.println("Path completed");
    }
}

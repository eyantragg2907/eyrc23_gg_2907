#include <WiFi.h>

const char *ssid = "pjrWifi";          // Enter your wifi hotspot ssid
const char *password = "SimplePass01"; // Enter your wifi hotspot password
const uint16_t port = 8002;
const char *host = "192.168.54.144";

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

int turntime = 750; // in milliseconds

int input1, input2, input3, input4, input5; // inputs of IR sensors, returns HIGH when black line

bool mode = 0; // 0 is bang bang, 1 is the middle line
bool node = true;

// String path = "nnnrlrrnrnln"; // n means do nothing, l = left, r = right
String path = "";

char incomingPacket[80];
WiFiClient client;

String msg = "0";

int i = 0; // flags
int operation = 0;
// 0 for forward, 1 for "on node now check command", 2 for rotating left, 3 for rotating right, 4 for leaving the node, 5 terminating
int rotflag = 0;
void stop()
{
    analogWrite(motor1f, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1r, 0);
    analogWrite(motor2r, 0);
}

int moveforwardtillreachnode()
{
    if (!(input2 == 1 && input3 == 1 && input4 == 1)) // if not at a node
    {
        if (input2 == 0 && input3 == 0 && input4 == 0) // bang bang controller
        {
            if (input1 == 1 && input5 == 0) // left line detected by left sensor
            {
                analogWrite(motor1f, speed1);
                analogWrite(motor2f, 0);
            }
            else if (input5 == 1 && input1 == 0) // right line detected by  right sensor
            {
                analogWrite(motor1f, 0);
                analogWrite(motor2f, speed2);
            }
            else
            {
                analogWrite(motor1f, speed1);
                analogWrite(motor2f, speed2);
            }
        }
        else if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
        {
            analogWrite(motor1f, speed1);
            analogWrite(motor2f, speed2);
        }
        else if (input2 == 1 && input4 == 0) // middle line detected by middle left sensor
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, speed2);
        }
        else if (input4 == 1 && input2 == 0) // middle line detected by middle right sensor
        {
            analogWrite(motor1f, speed1);
            analogWrite(motor2f, 0);
        }
        return 0;
    }
    else
    { // reached a node for sure
        stop();
        return 1;
    }
}

// int turn(int mode)
// {
//   stop();
//   if (mode == 1)
//   {
//     analogWrite(motor1r, speed1);
//     analogWrite(motor2f, speed2);
//     analogWrite(motor2r, 0);
//     analogWrite(motor1f, 0);
//   }
//   else
//   {
//     analogWrite(motor1r, 0);
//     analogWrite(motor2f, 0);
//     analogWrite(motor1f, speed1);
//     analogWrite(motor2r, speed2);
//   }
//   delay(turntime);
//   stop();
//   return 1;
// }

int turn(int mode)
{
    int input234_binary = 0x0;
    input234_binary |= input2 << 2;
    input234_binary |= input3 << 1;
    input234_binary |= input4 << 0;
    if (node)
    { // at a node
        analogWrite(motor1f, speed1);
        analogWrite(motor2f, speed2);
        delay(600);
        node = false;
    } // Now we have left the node for sure!
    else
    {
        if (rotflag == 0) // rotate a little bit to leave the middle black line
        {
            if (mode == 1) // rotate right
            {
                analogWrite(motor1r, speed1);
                analogWrite(motor2f, speed2);
                analogWrite(motor2r, 0);
                analogWrite(motor1f, 0);
            }
            else // rotate left
            {
                analogWrite(motor1r, 0);
                analogWrite(motor2f, 0);
                analogWrite(motor1f, speed1);
                analogWrite(motor2r, speed2);
            }
            delay(300);
            rotflag = 1;
            Serial.println("Rotated for 300ms to leave the middle black line!");
        }
        else if ((input234_binary == 0b110 || input234_binary == 0b011 || input234_binary == 0b010) && (input5 == 0 && input1 == 0)) // reached the middle line again, we completed rotation
        {
            stop();
            // query IRs, for checking after momentum
            int temp2 = digitalRead(IR2);
            int temp3 = digitalRead(IR3);
            int temp4 = digitalRead(IR4);
            input234_binary = 0x0;
            input234_binary |= temp2 << 2;
            input234_binary |= temp3 << 1;
            input234_binary |= temp4 << 0;
            // now if its okay, i can say we have reached line
            if (input234_binary == 0b010) {
                Serial.println("Rotation Completed");
                rotflag = 0;
                delay(3000);
                node = true;
                return 1;
            }
        }
        else if (mode == 1) // rotate right
        {
            analogWrite(motor1r, speed1);
            analogWrite(motor2f, speed2);
            analogWrite(motor2r, 0);
            analogWrite(motor1f, 0);
        }
        else // rotate left
        {
            analogWrite(motor1r, 0);
            analogWrite(motor2f, 0);
            analogWrite(motor1f, speed1);
            analogWrite(motor2r, speed2);
        }
    }
    return 0;
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

    client.print("Obese American ate 69 giant ramen bowl but still is lighter than your mom"); // Send an acknowledgement to host(laptop)
    msg = client.readStringUntil('\n');                                                        // Read the message through the socket until new line char(\n)
    path = msg;
    // move = true;
    Serial.println(path);
    Serial.println("Starting the run!");
    digitalWrite(led_red, LOW);
    digitalWrite(led_green, HIGH);
}

void loop()
{
    input1 = digitalRead(IR1); // read IR sensors
    input2 = digitalRead(IR2);
    input3 = digitalRead(IR3);
    input4 = digitalRead(IR4);
    input5 = digitalRead(IR5);
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
    if (operation == 0) // move forward
    {
        if (moveforwardtillreachnode())
        {
            operation = 6;
        }
    }
    else if (operation == 1) // Next Command
    {
        char command = path[i++];
        if (command == 'l')
        {
            Serial.println("Command to rotate left");
            operation = 2;
        }
        else if (command == 'r')
        {
            Serial.println("Command to rotate right");
            operation = 3;
        }
        else
        {
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
            analogWrite(motor1f, speed1);
            analogWrite(motor2f, speed2);
            delay(100);
        }
        else if (i == path.length())
        {
            stop();
            Serial.println("path complete, starting the terminating sequence ");
            operation = 5;
        }
        else
        {
            stop();
            operation = 0;
        }
    }
    else if (operation == 5) // terminate
    {
        Serial.println("Going to the ending node!");
        if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
        {
            analogWrite(motor1f, speed1);
            analogWrite(motor2f, speed2);
        }
        else if (input2 == 1 && input4 == 0) // middle line detected by middle left sensor
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, speed2);
        }
        else if (input4 == 1 && input2 == 0) // middle line detected by middle right sensor
        {
            analogWrite(motor1f, speed1);
            analogWrite(motor2f, 0);
        }
        else if (input3 == 0 && input2 == 0 && input4 == 0) // stop sign reached
        {
            // starting victory sequence
            Serial.println("Reached the ending node!");
            stop();
            digitalWrite(led_green, LOW);
            digitalWrite(led_red, HIGH);
            digitalWrite(buzzer, LOW);
            delay(5000);
            digitalWrite(led_red, LOW);
            digitalWrite(buzzer, HIGH);
            operation = 7;
        }
    }
    else if (operation == 6) // Node Found now what to do
    {
        Serial.println("NODE FOUND!! Stopping all engines!"); // starting the node reaching ritual
        stop();
        digitalWrite(buzzer, LOW);
        delay(1000);
        digitalWrite(buzzer, HIGH);
        operation = 1;
    }
    else
    {
        Serial.println("WE HAVE DONE IT!");
    }
}

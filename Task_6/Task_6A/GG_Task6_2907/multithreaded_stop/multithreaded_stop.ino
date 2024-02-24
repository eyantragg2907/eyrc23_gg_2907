/*
 * Team Id: 2907
 * Author List: Abhinav Lodha, Pranjal Rastogi
 * Filename: <Filename>
 * Theme: GeoGuide
 * Functions: stop, turn_right, turn_left, turn_right_faster, turn_left_faster, readIRs, printIRs, loop,
 * establishInitialWifiConnection, listenAndDirectActions, moveForwardLogic, moveForwardLogicSpecial, moveForwardTillReachedNode,
 * turn_special,setup,controlLoop, conductMovement
 * Global Variables: SPEED_LEFTMOTOR, SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR_SLOW, SPEED_RIGHTMOTOR_SLOW, SPEED_LEFTMOTOR_SLOWSLOW,
 * SPEED_RIGHTMOTOR_SLOWSLOW, ROTATE_SPEED, ROTATE_SPEED_LEFT, ROTATE_SPEED_UTURN, BANGBANG_TURNSPEED, MIDDLE_TURNSPEED,
 * ROT_COMPLETE_DELAY, EVENT_NODE_REACHED_DELAY, NORMAL_NODE_REACHED_DELAY, END_DELAY, CENTER_CORRECT_DELAY, LEAVE_BLACK_DELAY,
 * LEAVE_BLACK_DELAY_LEFT, UTURN_TIME, NODE_LEAVE_DELAY, IGNORE_FALSE_NODE_TIME, ALIGN_CENTER_BEGINNING, TURN_DELAY_BEGINNING,
 * ERROR_COUNTER_MAX, END_SKIP, END_SKIP_FORWARD_DELAY, CONNECTION_PING_DELAY, WIFI_TRY_DELAY, SETUP_DELAY, ssid, password, port,
 * host, IR1, IR2, IR3, IR4, IR5, motor1f, motor1r, motor2f, motor2r, led_red, led_green, buzzer, action_queue, reverse_action_queue,
 * message_queue, send_to_wifi_queue, MESSAGE_QUEUE_SIZE, START_ACTIONCODE, INTERRUPTSTOP_ACTIONCODE, LEFT, RIGHT, input1, input2,
 * input3, input4, input5, node_left_time, blackline_detect_start, rotflag, node, start_of_end_detect
 */

#include <WiFi.h>

/* Configuration */

/* SPEED CONTROLS */
#define SPEED_LEFTMOTOR 175  // motor LEFT speed, FORWARD
#define SPEED_RIGHTMOTOR 150 // down from 255 motor RIGHT speed, FORWARD
#define SPEED_RIGHTMOTOR_LO 130 // 'x' speed
#define SPEED_LEFTMOTOR_LO 130
#define SPEED_LEFTMOTOR_SLOW 75 // 'X' speed
#define SPEED_RIGHTMOTOR_SLOW 75
#define SPEED_LEFTMOTOR_SLOWSLOW 140 // 'd' speed
#define SPEED_RIGHTMOTOR_SLOWSLOW 140
#define ROTATE_SPEED 140 // motor BOTH speed, D90/ D180 TURNS
#define ROTATE_SPEED_LEFT 135 // or 70
#define ROTATE_SPEED_UTURN 200 // we turn faster on uturns since they are time-controlled anyway
#define XINCREASEBANGBANG 35

#define BANGBANG_TURNSPEED 220 // correction motor speed when in WALL mode
#define MIDDLE_TURNSPEED 125   // correction motor speed when in MIDDLE_LINE mode

/* STOP DELAYS */ 
#define ROT_COMPLETE_DELAY 100        // STOP delay after a D90 turn.
#define EVENT_NODE_REACHED_DELAY 1000 // STOP delay for every EVENT NODE. Also activates BUZZER
#define NORMAL_NODE_REACHED_DELAY 50  // STOP delay after every node. No BUZZER.
#define END_DELAY 5000                // delay for buzzer ring at the END

/* TURN LOGIC: DONT DO BLACK LINE DETECTION FOR X */
#define CENTER_CORRECT_DELAY 350   // delay to align center of rotation for turning 350
#define LEAVE_BLACK_DELAY 500      // delay before black line detection begins in turning D90 400
#define LEAVE_BLACK_DELAY_LEFT 590 // delay before black line detection begins in turning D90 for LEFT Turn

/* TURN LOGIC: UTURN  */
#define UTURN_TIME 845 // exact delay for which a D180 turn is undertaken. No black line detection happens in 180s.

/* FALSE NODE IGNORE LOGIC */
#define NODE_LEAVE_DELAY 140      // delay to move in front of a NODE w/o stopping logic. Without moving logic.
#define IGNORE_FALSE_NODE_TIME 60 // delay before node-detection logic fires up again. Counted after NODE_LEAVE_DELAY. With moving logic.

/* SPECIAL BEGINNING LOGIC */
#define ALIGN_CENTER_BEGINNING 90 // d: 80 // delay for aligning center of rotation in the beginning, when the situation is different.
#define TURN_DELAY_BEGINNING 30   // d: 50 // delay for a small left turn in the beginning, for correction purposes.

/* SPECIAL END LOGIC */
#define ERROR_COUNTER_MAX 11        // delay of the number of times false detection of ALL OFF can happen at the end.
#define END_SKIP 1200               // delay before END (ALL OFF) detection logic starts working. But with moving logic.
#define END_SKIP_FORWARD_DELAY 500 // delay for which simple forward movement is present in END detection
#define NODE_LEAVE_DELAY_END 80

/* WIFI AND SETUP */
#define CONNECTION_PING_DELAY 200 // delay between WIFI-host retry's
#define WIFI_TRY_DELAY 500        // delay between WIFI-connect retry's

#define SETUP_DELAY 8000 // delay in the beginning before the robot starts moving to give us time to put it on the grid

/* wireless */
const char *ssid = "brainerd";
const char *password = "internetaccess";
const uint16_t port = 8002;
const char *host = "192.168.67.62";
// laptops IP Address
// const char *ssid = "pjrWifi";
// const char *password = "SimplePass01";
// const uint16_t port = 8002;
// const char *host = "192.168.76.62"; // laptops IP Address
WiFiClient client;

/* IR sensor pins */
const int IR1 = 5;
const int IR2 = 25;
const int IR3 = 32;
const int IR4 = 33;
const int IR5 = 18;

/* motors */
const int motor1f = 13; // motor LEFT forward
const int motor1r = 27; // moto r LEFT reverse
const int motor2f = 12; // motor RIGHT forward
const int motor2r = 22; // motor RIGHT reverse, used to be 14

/* misc */
const int led_red = 2;
const int led_green = 15;
const int buzzer = 23;

/* multithreading */
QueueHandle_t action_queue;
QueueHandle_t reverse_action_queue;
QueueHandle_t message_queue;
QueueHandle_t send_to_wifi_queue;

#define MESSAGE_QUEUE_SIZE 40

/* enum constants */
#define START_ACTIONCODE 101
#define INTERRUPTSTOP_ACTIONCODE 103
#define LEFT 'L'  // Left.
#define RIGHT 'R' // Right

/* state variables */
int input1, input2, input3, input4, input5;
unsigned long node_left_time = 0;
unsigned long blackline_detect_start = 0;
int rotflag = 0;
bool node = true;
unsigned long start_of_end_detect = 0;

/*
 * Function Name: setup
 * Input: void
 * Output: void
 * Logic: Initializes the serial communication, IR sensor pins, motor pins, LED pins, buzzer pin, and the queues for multithreading
 * Example Call: Program calls this function automatically
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
    pinMode(led_green, OUTPUT);
    pinMode(buzzer, OUTPUT);

    digitalWrite(buzzer, HIGH); // switch it off!

    action_queue = xQueueCreate(10, sizeof(int));

    if (action_queue == NULL)
    {
        Serial.println("Error creating action_queue");
    }

    message_queue = xQueueCreate(10, sizeof(char) * MESSAGE_QUEUE_SIZE);

    if (message_queue == NULL)
    {
        Serial.print("Error creating message_queue");
    }

    reverse_action_queue = xQueueCreate(10, sizeof(int));
    if (reverse_action_queue == NULL)
    {
        Serial.println("Error creating reverse_action_queue");
    }

    send_to_wifi_queue = xQueueCreate(20, sizeof(char) * MESSAGE_QUEUE_SIZE);
    if (send_to_wifi_queue == NULL)
    {
        Serial.println("Error creating send_to_wifi_queue");
    }

    digitalWrite(led_red, HIGH);

    xTaskCreatePinnedToCore(controlLoop, "controlLoop", 8192, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(wifiLoop, "wifiLoop", 8192, NULL, 1, NULL, 1);
}

/*
 * Function Name: controlLoop
 * Input: void
 * Output: void
 * Logic: This function is the main control loop of the robot. It listens to the action_queue and performs the actions accordingly.
 * Example Call: Program calls this function automatically
 */
void controlLoop(void *pvParameters)
{
    while (1)
    {
        unsigned int action_code = 0;
        if (xQueueReceive(action_queue, &action_code, 0))
        {
            switch (action_code)
            {
            case START_ACTIONCODE:
            {
                // standard start-up procedure should be followed
                delay(SETUP_DELAY); // delay so that we have time to set up

                resetGlobals();
                digitalWrite(led_red, LOW);

                // read path
                char path[MESSAGE_QUEUE_SIZE] = "";
                if (xQueueReceive(message_queue, path, 0))
                {
                    Serial.print("Path message received succesfully.");
                }

                conductMovement(path);
            }
            default:
            {
                break;
            }
            }
        }
        delay(1);
    }
}

/*
 * Function Name: wifiLoop
 * Input: void *pvParameters -> Pointer that will be used as the parameter for the task being created
 * Output: void
 * Logic: This function is the main wifi loop of the robot.
 * It establishes the initial wifi connection and listens to the host for actions.
 * Example Call:  xTaskCreatePinnedToCore(wifiLoop, "wifiLoop", 8192, NULL, 1, NULL, 1);
 */
void wifiLoop(void *pvParameters)
{
    establishInitialWifiConnection(); // establishes the initial wifi connection
    listenAndDirectActions();         // listens to the host for actions
}

/*
 * Function Name: conductMovement
 * Input: char *path -> The path that the robot has to follow
 * Output: void
 * Logic: This function is the main control loop of the robot.
 * It listens to the action_queue and performs the actions accordingly.
 * Example Call: conductMovement("nnnxnnn");
 */
void conductMovement(char *path)
{
    readIRs();

    int path_len = strlen(path);
    char next_movement;
    Serial.print("Path length: ");
    Serial.println(path_len);
    
    for (int nc = 0; nc < path_len; nc++)
    {
        next_movement = path[nc];

        if (nc == 0 && next_movement == 'n')
        {
            // initial starting node
            Serial.println("===> Initial starting node.");
            do
            {
                readIRs();
            } while (!moveForwardTillReachedNode(SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR));

            analogWrite(motor1f, SPEED_LEFTMOTOR);
            analogWrite(motor2f, SPEED_RIGHTMOTOR);
            delay(ALIGN_CENTER_BEGINNING);
            stop();
            turn_left();
            delay(TURN_DELAY_BEGINNING);
            stop();

            // digitalWrite(buzzer, LOW);
            delay(NORMAL_NODE_REACHED_DELAY);
            // digitalWrite(buzzer, HIGH);

            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "left first node: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);

            node_left_time = millis();
        }
        else if (next_movement == 'l')
        {
            Serial.println("===> left.");
            do
            {
                readIRs();
            } while (!turn(LEFT, LEAVE_BLACK_DELAY_LEFT));
            stop();
            // turn_right();
            // delay(CORRECT_LEFT_DELAY);
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "90d left turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
            // delay(1000);
        }
        else if (next_movement == 'r')
        {
            Serial.println("===> right.");
            do
            {
                readIRs();
            } while (!turn(RIGHT, LEAVE_BLACK_DELAY));
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "90d right turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
        }
        else if (next_movement == 'n')
        {
            Serial.println("===> normal forward.");
            analogWrite(motor1r, 0);
            analogWrite(motor2r, 0);
            analogWrite(motor1f, SPEED_LEFTMOTOR);
            analogWrite(motor2f, SPEED_RIGHTMOTOR);
            delay(NODE_LEAVE_DELAY);
            node_left_time = millis();
            do
            {
                readIRs();
            } while (!moveForwardTillReachedNode(SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR));

            // digitalWrite(buzzer, LOW);
            delay(NORMAL_NODE_REACHED_DELAY);
            // digitalWrite(buzzer, HIGH);

            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "normal node left: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);

            node_left_time = millis();
        }
        else if (next_movement == 'd')
        {
            Serial.println("===> slow forward.");
            analogWrite(motor1r, 0);
            analogWrite(motor2r, 0);
            analogWrite(motor1f, SPEED_LEFTMOTOR);
            analogWrite(motor2f, SPEED_RIGHTMOTOR);
            delay(NODE_LEAVE_DELAY);
            node_left_time = millis();
            do
            {
                readIRs();
            } while (!moveForwardTillReachedNode(SPEED_RIGHTMOTOR_SLOWSLOW, SPEED_LEFTMOTOR_SLOWSLOW));

            // digitalWrite(buzzer, LOW);
            delay(NORMAL_NODE_REACHED_DELAY);
            // digitalWrite(buzzer, HIGH);

            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "normal node left: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);

            node_left_time = millis();
        }
        else if (next_movement == 'x')
        {
            Serial.println("===> special forward.");
            int send_me_istop = 1;
            if (xQueueSend(reverse_action_queue, &send_me_istop, portMAX_DELAY))
            {
              
                // analogWrite(motor1r, 0);
                // analogWrite(motor2r, 0);
                // analogWrite(motor1f, SPEED_LEFTMOTOR);
                // analogWrite(motor2f, SPEED_RIGHTMOTOR);
                // delay(NODE_LEAVE_DELAY);
                do
                {
                    readIRs();
                } while (!moveForwardTillStopped(SPEED_RIGHTMOTOR_LO, SPEED_LEFTMOTOR_LO));

                digitalWrite(buzzer, LOW);
                delay(EVENT_NODE_REACHED_DELAY);
                digitalWrite(buzzer, HIGH);

                char msg[MESSAGE_QUEUE_SIZE];
                snprintf(msg, MESSAGE_QUEUE_SIZE, "special node left: %lu\n", millis());
                xQueueSend(send_to_wifi_queue, msg, 0);

                node_left_time = millis();
            }
        }
        else if (next_movement == 'X')
        {
            Serial.println("===> special forward.");
            int send_me_istop = 1;
            if (xQueueSend(reverse_action_queue, &send_me_istop, portMAX_DELAY))
            {
                // analogWrite(motor1r, 0);
                // analogWrite(motor2r, 0);
                // analogWrite(motor1f, SPEED_LEFTMOTOR);
                // analogWrite(motor2f, SPEED_RIGHTMOTOR);
                // delay(NODE_LEAVE_DELAY);

                do
                {
                    readIRs();
                } while (!moveForwardTillStoppedSpecial(SPEED_RIGHTMOTOR_SLOW, SPEED_LEFTMOTOR_SLOW));

                digitalWrite(buzzer, LOW);
                delay(EVENT_NODE_REACHED_DELAY);
                digitalWrite(buzzer, HIGH);

                char msg[MESSAGE_QUEUE_SIZE];
                snprintf(msg, MESSAGE_QUEUE_SIZE, "special node left: %lu\n", millis());
                xQueueSend(send_to_wifi_queue, msg, 0);

                node_left_time = millis();
            }
        }
        else if (next_movement == 'R')
        {
            Serial.println("===> 180D TURN!");
            turn_right_faster();
            delay(UTURN_TIME);
            stop();
            readIRs();
            // do
            // {
            //     readIRs();
            // } while (!turn(RIGHT, LEAVE_BLACK_DELAY_UTURN));
            // Serial.println("===> right about turn.");
            // do
            // {
            //     readIRs();
            // } while (!turn(RIGHT, LEAVE_BLACK_DELAY));
            // node = false;
            // do
            // {
            //     readIRs();
            //     //} while (!turn(RIGHT, LEAVE_BLACK_DELAY - ABOUTTURN_SKIP_REDUCTION, BLACKLINE_INITIAL_SKIP - ABOUTTURN_SKIP_REDUCTION));
            // } while (!turn(RIGHT, 50));

            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "180d right turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
        }
        else if (next_movement == 'L')
        {
            Serial.println("===> 180D TURN!");
            // do
            // {
            //     readIRs();
            // } while (!turn(LEFT, LEAVE_BLACK_DELAY_UTURN));
            turn_left_faster();
            delay(UTURN_TIME);
            stop();
            readIRs();
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "180d left turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
        }
        else if (next_movement == 'p')
        {
            do
            {
                readIRs();
            } while (!turn_special(LEFT, LEAVE_BLACK_DELAY - 150));
        }
        else
        {
            // never reach here
        }
    }

    Serial.println("Terminating movement begins...");

    analogWrite(motor1r, 0);
    analogWrite(motor2r, 0);
    analogWrite(motor1f, SPEED_LEFTMOTOR);
    analogWrite(motor2f, SPEED_RIGHTMOTOR);
    delay(NODE_LEAVE_DELAY_END);

    start_of_end_detect = millis();
    int error_counter = 0;
    while (1)
    {
        readIRs();
        moveForwardLogicSpecial(SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR);
        if (millis() - start_of_end_detect >= END_SKIP)
        {
            if (input3 == 0 && input2 == 0 && input4 == 0)
            {
                error_counter += 1;
                if (error_counter == ERROR_COUNTER_MAX)
                {
                    endDetectionCode();
                    break;
                }
            }
        }
    }

    char msg[MESSAGE_QUEUE_SIZE];
    snprintf(msg, MESSAGE_QUEUE_SIZE, "terminate\n");
    xQueueSend(send_to_wifi_queue, msg, 0);
}

void resetGlobals()
{
    node_left_time = 0;
    blackline_detect_start = 0;
    rotflag = 0;
    node = true;
    start_of_end_detect = 0;
}

void endDetectionCode()
{
    Serial.println("Reached the ending node!");
    stop();

    analogWrite(motor1r, 0);
    analogWrite(motor2r, 0);
    analogWrite(motor1f, SPEED_LEFTMOTOR);
    analogWrite(motor2f, SPEED_RIGHTMOTOR);
    delay(END_SKIP_FORWARD_DELAY);
    stop();

    digitalWrite(led_red, HIGH);
    digitalWrite(buzzer, LOW);
    delay(END_DELAY);
    digitalWrite(led_red, LOW);
    digitalWrite(buzzer, HIGH);
}

int moveForwardTillStopped(int right_speed, int left_speed)
{
    moveForwardLogic(right_speed, left_speed);

    int action_code = 0;
    if (xQueueReceive(action_queue, &action_code, 0))
    {
        if (action_code == INTERRUPTSTOP_ACTIONCODE)
        {
            stop();
            return 1;
        }
    }
    return 0;
}

int moveForwardTillStoppedSpecial(int right_speed, int left_speed)
{
    moveForwardLogicHighBangBang(right_speed, left_speed);

    int action_code = 0;
    if (xQueueReceive(action_queue, &action_code, 0))
    {
        if (action_code == INTERRUPTSTOP_ACTIONCODE)
        {
            stop();
            return 1;
        }
    }
    return 0;
}


int turn(char dirn, int leave_black_delay)
{
    if (node)
    { // at a node
        // moved forward to align center of rotation
        analogWrite(motor1f, SPEED_LEFTMOTOR);
        analogWrite(motor2f, SPEED_RIGHTMOTOR);
        delay(CENTER_CORRECT_DELAY);
        node = false;
        return 0;
    } // Now we have left the node for sure!

    // leave black line w/o active detection
    if (rotflag == 0)
    {
        if (dirn == RIGHT)
        {
            turn_right();
        }
        else if (dirn == LEFT) // left
        {
            turn_left();
        }
        delay(leave_black_delay);
        stop();
        Serial.println("Rotated to leave the middle black line!");
        // delay(1000);
        rotflag = 1;
        return 0;
    }

    if (dirn == RIGHT) // rotate right
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0)) // reached the middle line again, we completed rotation
        {
            // client.print("BLACK_LINE DETECTED\n");
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
    else if (dirn == LEFT) // rotate left
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0))
        {
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

    return 0;
}

int turn_special(char dirn, int leave_black_delay)
{
    if (node)
    { // at a node
        // moved forward to align center of rotation
        analogWrite(motor1f, SPEED_LEFTMOTOR);
        analogWrite(motor2f, SPEED_RIGHTMOTOR);
        delay(CENTER_CORRECT_DELAY);
        node = false;
        return 0;
    } // Now we have left the node for sure!

    // leave black line w/o active detection
    if (rotflag == 0)
    {
        if (dirn == RIGHT)
        {
            turn_right();
        }
        else if (dirn == LEFT) // left
        {
            turn_left();
        }
        delay(leave_black_delay);
        Serial.println("Rotated to leave the middle black line!");
        rotflag = 1;
        return 0;
    }

    if (dirn == RIGHT) // rotate right
    {
        if (input3 == 1 && (input2 == 0 || input4 == 0)) // reached the middle line again, we completed rotation
        {
            // client.print("BLACK_LINE DETECTED\n");
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
    else if (dirn == LEFT) // rotate left
    {
        if (input3 == 1 && (input2 == 0 || input4 == 0))
        {
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

    return 0;
}

int moveForwardTillReachedNode(int right_speed, int left_speed)
{
    if (!((input2 == 1 && input3 == 1) && input4 == 1))
    {
        // if not at a node
        moveForwardLogic(right_speed, left_speed);
        return 0;
    }
    else
    { // reached a node
        if (millis() - node_left_time < IGNORE_FALSE_NODE_TIME)
        {
            moveForwardLogic(right_speed, left_speed);
            return 0;
        }
        else
        {
            stop();
            return 1;
        }
    }
}
void moveForwardLogic(int right_speed, int left_speed)
{

    // if (input1 == 1 && input5 == 0) // left line detected by left sensor
    //     {
    //         analogWrite(motor1f, BANGBANG_TURNSPEED);
    //         analogWrite(motor2f, 0);
    //     }
    //     else if (input5 == 1 && input1 == 0) // right line detected by right sensor
    //     {
    //         analogWrite(motor1f, 0);
    //         analogWrite(motor2f, BANGBANG_TURNSPEED);
    //     }
    //     else
    //     {
    //         analogWrite(motor1f, SPEED_LEFTMOTOR);
    //         analogWrite(motor2f, SPEED_RIGHTMOTOR);
    //     }
    //     return;
    if (input2 == 0 && input3 == 0 && input4 == 0) // bang bang controller
    {

        if ((input1 == 0 && input5 == 0) || (input1 == 1 && input5 == 1))
        {
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
        }
        else if (input1 == 1 && input5 == 0) // left line detected by left sensor
        {
            analogWrite(motor1f, BANGBANG_TURNSPEED);
            analogWrite(motor2f, 0);
        }
        else if (input5 == 1 && input1 == 0) // right line detected by right sensor
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, BANGBANG_TURNSPEED);
        }
    }
    else
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
        {
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
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

void moveForwardLogicHighBangBang(int right_speed, int left_speed)
{
    if (input2 == 0 && input3 == 0 && input4 == 0) // bang bang controller
    {

        if ((input1 == 0 && input5 == 0) || (input1 == 1 && input5 == 1))
        {
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
        }
        else if (input1 == 1 && input5 == 0) // left line detected by left sensor
        {
            analogWrite(motor1f, BANGBANG_TURNSPEED + XINCREASEBANGBANG);
            analogWrite(motor2f, 0);
        }
        else if (input5 == 1 && input1 == 0) // right line detected by right sensor
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, BANGBANG_TURNSPEED + XINCREASEBANGBANG);
        }
    }
    else
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
        {
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
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

void moveForwardLogicSpecial(int right_speed, int left_speed)
{
    if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
    {
        analogWrite(motor1f, left_speed);
        analogWrite(motor2f, right_speed);
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

/* wifi functions */
void establishInitialWifiConnection()
{
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(WIFI_TRY_DELAY);
        Serial.print("Attempting... ");
        Serial.println(ssid);
    }

    Serial.print("WiFi connected with IP: ");
    Serial.println(WiFi.localIP());

    do
    {
        Serial.println(host);
        Serial.println("Connection to host failed... Retrying.");
        delay(CONNECTION_PING_DELAY);
    } while (!client.connect(host, port));

    client.print("ACK_REQ_FROM_ROBOT");
    Serial.println("Connection to host successful");
}

void listenAndDirectActions()
{
    while (1)
    {
        unsigned int action_code = 0;

        String clientMessage = client.readStringUntil('\n');
        if (clientMessage.length() != 0)
        {
            if (clientMessage == "START")
            {
                // The two messages after the "START" message
                char message[MESSAGE_QUEUE_SIZE];
                String path = client.readStringUntil('\n');

                path.toCharArray(message, MESSAGE_QUEUE_SIZE);

                if (xQueueSend(message_queue, message, portMAX_DELAY))
                {
                    Serial.println("SENT MESSAGE TO MESSAGE_QUEUE");
                }

                action_code = START_ACTIONCODE;
                if (xQueueSend(action_queue, &action_code, portMAX_DELAY))
                {
                    Serial.println("SENT START_ACTION TO ACTION_QUEUE");
                }
            }
            else if (clientMessage == "ISTOP")
            {
                /* this thread can be flooded with too many ISTOP messages, but it should not flood the queue, and the messages are just unhandled (dropped)*/
                action_code = INTERRUPTSTOP_ACTIONCODE;
                int to_send = 0;
                if (xQueueReceive(reverse_action_queue, &to_send, 0))
                {
                    if (to_send)
                    {
                        if (xQueueSend(action_queue, &action_code, portMAX_DELAY))
                        {
                            Serial.println("SENT INTERRUPT-STOP TO ACTION_QUEUE");
                        }
                    }
                }
            }
        }

        char msg_send[MESSAGE_QUEUE_SIZE];
        if (xQueueReceive(send_to_wifi_queue, msg_send, 0))
        {
            client.print(msg_send);
        }
        delay(1);
    }
}

/* elementary functions */
void stop()
{
    analogWrite(motor1f, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1r, 0);
    analogWrite(motor2r, 0);
}
void turn_right()
{
    analogWrite(motor1r, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1f, ROTATE_SPEED);
    analogWrite(motor2r, ROTATE_SPEED);
}
void turn_left()
{
    analogWrite(motor2r, 0);
    analogWrite(motor1f, 0);
    analogWrite(motor1r, ROTATE_SPEED_LEFT + 10);
    analogWrite(motor2f, ROTATE_SPEED_LEFT - 25);
}

void turn_right_faster()
{
    analogWrite(motor1r, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1f, ROTATE_SPEED_UTURN);
    analogWrite(motor2r, ROTATE_SPEED_UTURN);
}
void turn_left_faster()
{
    analogWrite(motor2r, 0);
    analogWrite(motor1f, 0);
    analogWrite(motor1r, ROTATE_SPEED_UTURN);
    analogWrite(motor2f, ROTATE_SPEED_UTURN);
}
void readIRs()
{
    input1 = digitalRead(IR1);
    input2 = digitalRead(IR2);
    input3 = digitalRead(IR3);
    input4 = digitalRead(IR4);
    input5 = digitalRead(IR5);

    printIRs();
}
void printIRs()
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
}

void loop()
{
    // EMPTY (always)
}

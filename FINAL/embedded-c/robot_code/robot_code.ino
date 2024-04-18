/*
 * Team Id:                     2907
 * Author List:                 Abhinav Lodha, Pranjal Rastogi
 * Filename:                    robot_code.ino
 * Theme:                       GeoGuide
* Functions:                    setup(), controlLoop(void *), wifiLoop(void *), conductMovement(char *), endDetectionCode(), turn(char, int),
*                               moveForwardTillReachedNode(int, int), moveForwardTillStopped(int, int), moveForwardTillStoppedSpecial(int, int),
*                               moveForwardLogic(int, int), moveForwardLogicHighBangBang(int, int), moveForwardLogicSpecial(int, int),
*                               establishInitialWifiConnection(), listenAndDirectActions(), stop(), turn_left(), turn_right(), turn_right_faster(),
*                               turn_left_faster(), readIRs(), printIRs(), loop()
 * Global Variables: (multiple kinds:)
 *  - Defined Constants:        SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR, SPEED_RIGHTMOTOR_x, SPEED_LEFTMOTOR_x, SPEED_RIGHTMOTOR_e, SPEED_LEFTMOTOR_e,
 *                              SPEED_RIGHTMOTOR_d, SPEED_LEFTMOTOR_d, ROTATE_SPEED_RIGHT, ROTATE_SPEED_LEFT_L, ROTATE_SPEED_LEFT_R, ROTATE_SPEED_UTURN,
 *                              BANGBANG_TURNSPEED, XINCREASEBANGBANG, MIDDLE_TURNSPEED, ROT_COMPLETE_DELAY, EVENT_NODE_REACHED_DELAY, NORMAL_NODE_REACHED_DELAY,
 *                              END_DELAY, CENTER_CORRECT_DELAY, LEAVE_BLACK_DELAY_RIGHT, LEAVE_BLACK_DELAY_LEFT, p_LEAVE_BLACK_DELAY_REDUCTION, UTURN_TIME,
 *                              NODE_LEAVE_DELAY, IGNORE_FALSE_NODE_TIME, ALIGN_CENTER_BEGINNING, TURN_DELAY_BEGINNING, NODE_LEAVE_DELAY_END, ERROR_COUNTER_MAX,
 *                              END_SKIP, END_SKIP_FORWARD_DELAY, WIFI_TRY_DELAY, CONNECTION_PING_DELAY, SETUP_DELAY, MESSAGE_QUEUE_SIZE, START_ACTIONCODE, INTERRUPTSTOP_ACTIONCODE
 *                              LEFT, RIGHT
 * - Other variables:           ssid, password, port, host, client, IR1, IR2, IR3, IR4, IR5, motor1f, motor1r, motor2f, motor2r, led_red, buzzer, action_queue,
 *                              reverse_action_queue, message_queue, send_to_wifi_queue, input1, input2, input3, input4, input5, node_left_time, blackline_detect_start,
 *                              rotflag, node, start_of_end_detect
 */

#include <WiFi.h>

/* The following constants are for Robot Configuration and Speed Settings */

/* Speed Control Variables: Adjusting PWM Motor Speed */
#define SPEED_RIGHTMOTOR 140 // 135 // 140 // 145 // 130    // Standard Right Motor Speed.
#define SPEED_LEFTMOTOR 175     // Standard Left Motor Speed.
#define SPEED_RIGHTMOTOR_x 105 // 110 // 115 // 95 // Right Motor Speed when an EVENT_NODE needs to be detected.
#define SPEED_LEFTMOTOR_x 130   // Left Motor Speed when an EVENT_NODE needs to be detected.
#define SPEED_RIGHTMOTOR_e 95 // 105 // 110 // 85  // 95 // Right Motor Speed when an EVENT_NODE needs to be detected in certain areas.
#define SPEED_LEFTMOTOR_e 110 // 120 // 110   // Left Motor Speed when an EVENT_NODE needs to be detected in certain areas.
#define SPEED_RIGHTMOTOR_d 88 // 90 // 100  // 100 // Right Motor Speed in certain areas.
#define SPEED_LEFTMOTOR_d 105 // 120   // Left Motor Speed in certain areas.
#define ROTATE_SPEED_RIGHT 150  // Speed for both motors when doing a D90 Right turn.
#define ROTATE_SPEED_LEFT_L 125 // 135 // Speed for left motor (reverse) when doing a D90 Left turn.
#define ROTATE_SPEED_LEFT_R 100 // 105 // 110 // 100 // 110 // Speed for right motor (forward) when doing a D90 Left turn.
#define ROTATE_SPEED_SLOW_LEFT 65 // Speed for left turn at the end for special case.
#define ROTATE_SPEED_UTURN 200  // Speed for both motors in all kinds of D180 turns.

#define BANGBANG_TURNSPEED 235 // The speed for corrective movements in both directions in WALL mode.
#define XINCREASEBANGBANG 15  // An additive factor that increases BANGBANG_TURNSPEED when an EVENT_NODE needs to be detected.

#define MIDDLE_TURNSPEED 125 // The speed for corrective movements in both directions in MIDDLE_LINE mode.

/* Stop Delays: Controlling robot momentum */
#define ROT_COMPLETE_DELAY 100        // Stop delay after a D90 turn.
#define EVENT_NODE_REACHED_DELAY 1000 // Stop delay for every EVENT NODE. Also activates BUZZER.
#define NORMAL_NODE_REACHED_DELAY 100  // Stop delay after every node. No BUZZER.
#define END_DELAY 5000                // Stop delay for buzzer ring at the end.
#define DELAY_AFTER_UTURN 150

/* Turn Logic: Delays that control how turning works */
#define CENTER_CORRECT_DELAY 370          // Delay for which the motors run to align center of rotation for turning.
#define LEAVE_BLACK_DELAY_RIGHT 505       // Delay before black line detection begins while turning right.
#define LEAVE_BLACK_DELAY_LEFT 590        // Delay before black line detection begins while turning left.
#define p_LEAVE_BLACK_DELAY_REDUCTION 570 // Delay reduction for black line detection while turning left in certain areas.
#define UTURN_TIME 810 // 835 // 845                    // Exact delay to conduct a succesful D180 turn. No black line detection happens in u-turns.

/* Delays that control how False Node detection logic works */
#define NODE_LEAVE_DELAY 160      // Delay for which robot continues to move after reaching a Node, without any moving logic.
#define IGNORE_FALSE_NODE_TIME 60 // Delay for which robot continues to move after the above delay, with moving logic.

/* Delays to control how the robot moves in the beginning */
#define ALIGN_CENTER_BEGINNING 90 // 90 // Delay for which the motors run to align center of rotation in the beginning for turning.
#define TURN_DELAY_BEGINNING 30  // 30 // Delay for which a small left turn is conducted in the beginning, for correction purposes.

/* Delays to control how the robot moves at the end */
#define NODE_LEAVE_DELAY_END 80    // Delay for which robot continues to move before any of the terminating logic starts.
#define ERROR_COUNTER_MAX 11       // Number of times a false detection of "ALL OFF" can happen at the end.
#define END_SKIP 1200              // Delay before "ALL OFF" detection logic starts working. Robot moves with moving logic during this.
#define END_SKIP_FORWARD_DELAY 500 // Delay for which simple forward movement is present after "ALL OFF" detection.
#define SPECIAL_END_DELAY_FOR_p 700

/* WiFi and Others */
#define WIFI_TRY_DELAY 500        // Delay between WIFI-connect-to-network retry's.
#define CONNECTION_PING_DELAY 200 // Delay between WiFi-connect-to-host to host retry's.
#define SETUP_DELAY 0 // 8000 // 0 to no delay          // Delay in the beginning before the robot starts moving to give us time to put it on the arena.

/* Wifi Variables */
const char *ssid = "brainerd";
const char *password = "internetaccess";
const uint16_t port = 8002;
const char *host = "192.168.95.62";

WiFiClient client;

/* Configured Pins */

/* IR sensor pins */
const int IR1 = 5; // Left most sensor, when looking from the front.
const int IR2 = 25;
const int IR3 = 32;
const int IR4 = 33;
const int IR5 = 18; // Right most sensor, when looking from the front.

/* motors */
const int motor1f = 13; // Left Motor forward.
const int motor1r = 27; // Left Motor reverse.
const int motor2f = 12; // Right Motor forward.
const int motor2r = 22; // Right Motor reverse.

/* misc */
const int led_red = 2;
const int buzzer = 23;

/* multithreading queue set up */
QueueHandle_t action_queue;
QueueHandle_t reverse_action_queue;
QueueHandle_t message_queue;
QueueHandle_t send_to_wifi_queue;

#define MESSAGE_QUEUE_SIZE 60

/* Constants that are used as Enum values */
#define START_ACTIONCODE 101
#define INTERRUPTSTOP_ACTIONCODE 103
#define LEFT 'L'  // Represents Left while calling some internal functions.
#define RIGHT 'R' // Represents right while calling some internal functions.

/* State variables */
int input1, input2, input3, input4, input5;
unsigned long node_left_time = 0; // Used in several areas to keep track of time when a node was left.
unsigned long blackline_detect_start = 0;
int rotflag = 0;
bool node = true;
unsigned long start_of_end_detect = 0;

/*
 * Function Name:   setup
 * Input:           void
 * Output:          void
 * Logic:           Initializes the serial communication, IR sensor pins, motor pins, LED pins, buzzer pin, and the queues for multithreading
 * Example Call:    Program calls this function automatically
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

    digitalWrite(buzzer, HIGH); // switch it off!

    /* create queues for our threads: */
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

    /* start the threads */
    xTaskCreatePinnedToCore(controlLoop, "controlLoop", 8192, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(wifiLoop, "wifiLoop", 8192, NULL, 1, NULL, 1);
}

/*
 * Function Name:       controlLoop
 * Input:               void *pvParameters -> Array of parameters, unused
 * Output:              void
 * Logic:               This function is the main control loop of the robot. 
 *                      It listens to the action_queue and performs the actions accordingly.
 * Example Call:        Program calls this function in the thread using xTaskCreatePinnedToCore. 
 *                      Do not call this function manually.
 */
void controlLoop(void *pvParameters)
{
    while (1)
    {
        unsigned int action_code = 0;
        // if we received a action code from the queue, we need to start
        if (xQueueReceive(action_queue, &action_code, 0))
        {
            switch (action_code)
            {
            case START_ACTIONCODE:
            {
                // standard start-up procedure should be followed now

                delay(SETUP_DELAY);

                digitalWrite(led_red, LOW); // start of run!

                // read path from wifi
                char path[MESSAGE_QUEUE_SIZE] = "";
                if (xQueueReceive(message_queue, path, 0))
                {
                    Serial.print("Path message received succesfully.");
                }

                // start movement with path message
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
 * Function Name:           wifiLoop
 * Input:                   void *pvParameters -> Array of parameters, unused
 * Output:                  void
 * Logic:                   This function is the main wifi loop of the robot.
 *                          It establishes the initial wifi connection and listens to the host for actions.
 * Example Call:            Program calls this function in the thread using xTaskCreatePinnedToCore. 
 *                          Do not call this function manually.
 */
void wifiLoop(void *pvParameters)
{
    establishInitialWifiConnection();
    listenAndDirectActions();
}

/*
 * Function Name:   conductMovement
 * input:           char *path -> The path that the robot has to follow
 * Output:          void
 * Logic:           This function makes the robot move.
 *                  It goes through the path (given by the Python host) and follows instructions accordingly. Different letters in the path mean different things.
 *                  The meanings are:
 *                      - 'n': move to next node
 *                      - 'l': take a left turn
 *                      - 'r': take a right turn
 *                      - 'd': move to next node in certain situations
 *                      - 'x': move to the next event node
 *                      - 'X': move to the next event node in certain conditions
 *                      - 'R': take a u-turn, moving right
 *                      - 'L': take a u-turn, moving left (rarely used)
 *                      - 'p': take a left turn with altered conditions
 *
 * Example Call:    conductMovement("nnnxnnn");
 */
void conductMovement(char *path)
{
    readIRs();

    int path_len = strlen(path);
    char next_movement;
    Serial.print("Path length: ");
    Serial.println(path_len);

    bool over = false;

    for (int nc = 0; nc < path_len; nc++)
    {
        // for every character "next_movement" in path ...
        next_movement = path[nc];

        if (nc == 0 && next_movement == 'n') // We need to move till the initial starting node.
        {
            Serial.println("===> Initial starting node.");

            // move forwad till we reach the first node
            do
            {
                readIRs();
            } while (!moveForwardTillReachedNode(SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR));

            // special beginning node alignment
            analogWrite(motor1f, SPEED_LEFTMOTOR);
            analogWrite(motor2f, SPEED_RIGHTMOTOR);
            delay(ALIGN_CENTER_BEGINNING);
            stop();
            turn_left();
            delay(TURN_DELAY_BEGINNING);
            stop();

            delay(NORMAL_NODE_REACHED_DELAY);

            // send message to wifi for debug purposes
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "left first node: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);

            node_left_time = millis();
        }
        else if (next_movement == 'l') //  Take a left turn.
        {
            Serial.println("===> left.");

            // take the turn using the turn function.
            do
            {
                readIRs();
            } while (!turn(LEFT, LEAVE_BLACK_DELAY_LEFT));
            stop();

            // send message to wifi for debugging purposes
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "90d left turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
        }
        else if (next_movement == 'r') // Take a right turn.
        {
            Serial.println("===> right.");

            // take the turn using the turn function
            do
            {
                readIRs();
            } while (!turn(RIGHT, LEAVE_BLACK_DELAY_RIGHT));

            // send the message for debugging purposes
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "90d right turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
        }
        else if (next_movement == 'n') // We need to move till next node.
        {
            Serial.println("===> normal forward.");

            // Move a little to leave the previous node.
            analogWrite(motor1r, 0);
            analogWrite(motor2r, 0);
            analogWrite(motor1f, SPEED_LEFTMOTOR);
            analogWrite(motor2f, SPEED_RIGHTMOTOR);
            delay(NODE_LEAVE_DELAY);

            node_left_time = millis(); // keep track of time when we left the node

            // Move forward till we reach the next node
            do
            {
                readIRs();
            } while (!moveForwardTillReachedNode(SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR));

            delay(NORMAL_NODE_REACHED_DELAY);

            // send message to wifi for debugging purposes
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "normal node left: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);

            node_left_time = millis(); // reset
        }
        else if (next_movement == 'd') // We need to move forward till the next node under certain conditions.
        {
            Serial.println("===> slow forward.");

            // Move a little to leave the previous node.
            analogWrite(motor1r, 0);
            analogWrite(motor2r, 0);
            analogWrite(motor1f, SPEED_LEFTMOTOR);
            analogWrite(motor2f, SPEED_RIGHTMOTOR);

            delay(NODE_LEAVE_DELAY);

            // Move forward till we reach the next node, but with different speeds.
            node_left_time = millis();
            do
            {
                readIRs();
            } while (!moveForwardTillReachedNode(SPEED_RIGHTMOTOR_d, SPEED_LEFTMOTOR_d));

            delay(NORMAL_NODE_REACHED_DELAY);

            // send message to wifi for debugging purposes
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "normal node left: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);

            node_left_time = millis();
        }
        else if (next_movement == 'x') // We need to move forward till the next event node.
        {
            Serial.println("===> special forward.");
            int send_me_istop = 1;

            // send a notification to the wifi loop that we want to be "ISTOPped" when we reach the next event node.
            if (xQueueSend(reverse_action_queue, &send_me_istop, portMAX_DELAY))
            {
                // Move forward until we receive the "ISTOP" action in the queue.
                do
                {
                    readIRs();
                } while (!moveForwardTillStopped(SPEED_RIGHTMOTOR_x, SPEED_LEFTMOTOR_x));

                // Ring buzzer since event node!
                digitalWrite(buzzer, LOW);
                delay(EVENT_NODE_REACHED_DELAY);
                digitalWrite(buzzer, HIGH);

                // send message to wifi for debugging purposes
                char msg[MESSAGE_QUEUE_SIZE];
                snprintf(msg, MESSAGE_QUEUE_SIZE, "special node left: %lu\n", millis());
                xQueueSend(send_to_wifi_queue, msg, 0);

                node_left_time = millis();
            }
        }
        else if (next_movement == 'X') // Move forward till the next event node in special conditions.
        {
            Serial.println("===> special forward.");

            int send_me_istop = 1;
            // send a notification to the wifi loop that we want to be "ISTOPped" when we reach the next event node.
            if (xQueueSend(reverse_action_queue, &send_me_istop, portMAX_DELAY))
            {
                // Move forward with altered speeds until we receive the "ISTOP" action in the queue.
                do
                {
                    readIRs();
                } while (!moveForwardTillStoppedSpecial(SPEED_RIGHTMOTOR_e, SPEED_LEFTMOTOR_e));

                // Ring buzzer since event node!
                digitalWrite(buzzer, LOW);
                delay(EVENT_NODE_REACHED_DELAY);
                digitalWrite(buzzer, HIGH);

                // send message to wifi for debugging purposes
                char msg[MESSAGE_QUEUE_SIZE];
                snprintf(msg, MESSAGE_QUEUE_SIZE, "special node left: %lu\n", millis());
                xQueueSend(send_to_wifi_queue, msg, 0);

                node_left_time = millis();
            }
        }
        else if (next_movement == 'R') // Take a u-turn rightwards
        {
            Serial.println("===> 180D TURN!");

            // Turn right for a fixed time
            turn_right_faster();
            delay(UTURN_TIME);
            stop();
            delay(DELAY_AFTER_UTURN);
            readIRs();

            // debug message
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "180d right turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
        }
        else if (next_movement == 'L') // Take a u-turn leftwards
        {
            Serial.println("===> 180D TURN!");

            // Turn left for a fixed time
            turn_left_faster();
            delay(UTURN_TIME);
            stop();
            readIRs();

            // debug message
            char msg[MESSAGE_QUEUE_SIZE];
            snprintf(msg, MESSAGE_QUEUE_SIZE, "180d left turn complete: %lu\n", millis());
            xQueueSend(send_to_wifi_queue, msg, 0);
        }
        else if (next_movement == 'p') // Take a small left turn with altered conditions
        {
            // special left turn
            do
            {
                readIRs();
            } while (!turnForEndOnly(LEFT, LEAVE_BLACK_DELAY_LEFT - p_LEAVE_BLACK_DELAY_REDUCTION));

            analogWrite(motor1r, 0);
            analogWrite(motor2r, 0);
            analogWrite(motor1f, SPEED_LEFTMOTOR);
            analogWrite(motor2f, SPEED_RIGHTMOTOR);
            delay(SPECIAL_END_DELAY_FOR_p);

            stop();

            over = true;
        }
        else
        {
            // never reach here, otherwise it's a serious issue!
        }
    }

    if (over == false) {
        // After the path is complete, we move forward till the end of the arena ("ALL OFF").
        Serial.println("Terminating movement begins...");

        // Move a little forward to avoid false detections
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
            // Keep moving forward till we reach the end of the arena
            moveForwardLogicSpecial(SPEED_RIGHTMOTOR, SPEED_LEFTMOTOR);
            if (millis() - start_of_end_detect >= END_SKIP)
            {
                // If we have moved forward enough and want to detect the "ALL OFF" condition
                if (input3 == 0 && input2 == 0 && input4 == 0)
                {
                    // If we have reached the "ALL OFF" condition and have done it ERROR_COUNTER_MAX times,
                    // we consider that we have reached the end of the arena.
                    error_counter += 1;
                    if (error_counter == ERROR_COUNTER_MAX)
                    {
                        endDetectionCode();
                        break;
                    }
                }
            }
        }

        // end of movement!
        char msg[MESSAGE_QUEUE_SIZE];
        snprintf(msg, MESSAGE_QUEUE_SIZE, "terminate\n");
        xQueueSend(send_to_wifi_queue, msg, 0);
    }
    else {
        digitalWrite(led_red, HIGH);
        digitalWrite(buzzer, LOW);
        // digitalWrite(buzzer, HIGH); // comment this if u want buzzer.
        delay(END_DELAY);
        digitalWrite(led_red, LOW);
        digitalWrite(buzzer, HIGH);
        // end of movement!
        char msg[MESSAGE_QUEUE_SIZE];
        snprintf(msg, MESSAGE_QUEUE_SIZE, "terminate\n");
        xQueueSend(send_to_wifi_queue, msg, 0);
    }
}

/*
 * Function Name:       endDetectionCode
 * input:               void
 * Output:              void
 * Logic:               Special function that is called after the "ALL OFF" condition is met.
 *                      It moves the robot a little more, and then rings the buzzer.
 * Example Call:        endDetectionCode();
 */
void endDetectionCode()
{
    Serial.println("Reached the ending node!");
    stop();

    // move forward a little more
    analogWrite(motor1r, 0);
    analogWrite(motor2r, 0);
    analogWrite(motor1f, SPEED_LEFTMOTOR);
    analogWrite(motor2f, SPEED_RIGHTMOTOR);
    delay(END_SKIP_FORWARD_DELAY);
    stop();

    // ring the buzzer
    digitalWrite(led_red, HIGH);
    digitalWrite(buzzer, LOW);
    // digitalWrite(buzzer, HIGH); // comment this if u want buzzer.
    delay(END_DELAY);
    digitalWrite(led_red, LOW);
    digitalWrite(buzzer, HIGH);
}

/*
 * Function Name:       turn
 * Input:               char dirn, int leave_black_delay
 * Output:              int return_code, 0 if turn is not complete, 1 if it is
 * Logic:               Turns in "dirn" direction while ignoring the black 
 *                      line for "leave_black_delay".
 * Example Call:        turn(LEFT, 530)
 */
int turn(char dirn, int leave_black_delay)
{
    if (node)
    {
        // at a node, move a little forward to align center of rotation
        analogWrite(motor1f, SPEED_LEFTMOTOR);
        analogWrite(motor2f, SPEED_RIGHTMOTOR);
        delay(CENTER_CORRECT_DELAY);
        node = false;
        return 0;
    } // Now we have left the node for sure!

    // rotflag = 0 indicates that we have not yet rotated to leave the middle black line.
    // rotate for fixed time to leave the middle black line
    if (rotflag == 0)
    {
        if (dirn == RIGHT)
        {
            turn_right();
        }
        else if (dirn == LEFT)
        {
            turn_left();
        }
        delay(leave_black_delay);
        stop();
        Serial.println("Rotated to leave the middle black line!");
        rotflag = 1;
        return 0;
    }

    // rotflag != 0 indicates that we have rotated to leave the middle black line.
    // now we have to rotate till we reach the middle line again.
    if (dirn == RIGHT)
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0))
        {
            // reached the middle line again, we completed rotation

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
    else if (dirn == LEFT)
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0))
        {
            // reached the middle line again, we completed rotation

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

/*
 * Function Name:       turn
 * Input:               char dirn, int leave_black_delay
 * Output:              int return_code, 0 if turn is not complete, 1 if it is
 * Logic:               Turns in "dirn" direction while ignoring the black 
 *                      line for "leave_black_delay".
 * Example Call:        turn(LEFT, 530)
 */
int turnForEndOnly(char dirn, int leave_black_delay)
{
    if (node)
    {
        // at a node, move a little forward to align center of rotation
        analogWrite(motor1f, SPEED_LEFTMOTOR);
        analogWrite(motor2f, SPEED_RIGHTMOTOR);
        delay(CENTER_CORRECT_DELAY);
        node = false;
        return 0;
    } // Now we have left the node for sure!

    // rotflag = 0 indicates that we have not yet rotated to leave the middle black line.
    // rotate for fixed time to leave the middle black line
    if (rotflag == 0)
    {
        if (dirn == RIGHT)
        {
            turn_right();
        }
        else if (dirn == LEFT)
        {
            turn_left();
        }
        delay(leave_black_delay);
        stop();
        Serial.println("Rotated to leave the middle black line!");
        rotflag = 1;
        return 0;
    }

    // rotflag != 0 indicates that we have rotated to leave the middle black line.
    // now we have to rotate till we reach the middle line again.
    if (dirn == RIGHT)
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0))
        {
            // reached the middle line again, we completed rotation

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
    else if (dirn == LEFT)
    {
        if (input3 == 1 && (input2 == 0 && input4 == 0))
        {
            // reached the middle line again, we completed rotation

            Serial.println("Rotation Completed");
            rotflag = 0;
            node = true;
            stop();
            delay(ROT_COMPLETE_DELAY);
            return 1;
        }
        else
        {
            turn_left_slower();
        }
    }

    return 0;
}


/*
 * Function Name:       moveForwardTillReachedNode
 * Input:               int right_speed, int left_speed
 * Output:              int return_code, 0 if not at a node, 1 if at a node
 * Logic:               Moves forward using moveForwardLogic and stops when a node is detected.
 * Example Call:        moveForwardTillReachedNode(220, 220)
 */
int moveForwardTillReachedNode(int right_speed, int left_speed)
{
    if (!((input2 == 1 && input3 == 1) && input4 == 1))
    {
        // if not at a node, just move forward!
        moveForwardLogic(right_speed, left_speed);
        return 0;
    }
    else
    {
        // reached a node
        if (millis() - node_left_time < IGNORE_FALSE_NODE_TIME)
        {
            // if we are still within the ignore time limit, we ignore the node
            moveForwardLogic(right_speed, left_speed);
            return 0;
        }
        else
        {
            // node! return 1 to signify.
            stop();
            return 1;
        }
    }
}

/*
 * Function Name:       moveForwardTillStopped
 * Input:               int right_speed, int left_speed
 * Output:              int hasStopped, 1 if stopped, 0 if not
 * Logic:               Moves using moveForwardLogic and stops when an event node is reached.
 * Example Call:        moveForwardTillStopped(220, 220)
 */
int moveForwardTillStopped(int right_speed, int left_speed)
{
    moveForwardLogic(right_speed, left_speed);

    int action_code = 0;
    if (xQueueReceive(action_queue, &action_code, 0)) // if there is a ISTOP action in the queue, we stop.
    {
        if (action_code == INTERRUPTSTOP_ACTIONCODE)
        {
            // ISTOP was present in the queue, stop!
            stop();
            return 1;
        }
    }
    return 0;
}

/*
 * Function Name:       moveForwardTillStoppedSpecial
 * Input:               int right_speed, int left_speed
 * Output:              int hasStopped, 1 if stopped, 0 if not
 * Logic:               Moves using moveForwardLogicHighBangBang and stops when an event node is reached.
 * Example Call:        moveForwardTillStoppedSpecial(220, 220)
 */
int moveForwardTillStoppedSpecial(int right_speed, int left_speed)
{
    moveForwardLogicHighBangBang(right_speed, left_speed);

    int action_code = 0;
    if (xQueueReceive(action_queue, &action_code, 0)) // if there is a ISTOP action in the queue, we stop.
    {
        if (action_code == INTERRUPTSTOP_ACTIONCODE)
        {
            // ISTOP was present in the queue, stop!
            stop();
            return 1;
        }
    }
    return 0;
}

/*
 * Function Name:       moveForwardLogic
 * Input:               int right_speed, int left_speed
 * Output:              void
 * Logic:               Standard logic for moving forward.
 * Example Call:        moveForwardLogic(220, 220)
 */
void moveForwardLogic(int right_speed, int left_speed)
{
    if (input2 == 0 && input3 == 0 && input4 == 0) // bang bang controller for "wall mode"
    {

        if ((input1 == 0 && input5 == 0) || (input1 == 1 && input5 == 1)) // forward if no line detected or both lines detected
        {
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
        }
        else if (input1 == 1 && input5 == 0) // left line detected by left sensor, move left to align
        {
            analogWrite(motor1f, BANGBANG_TURNSPEED);
            analogWrite(motor2f, 0);
        }
        else if (input5 == 1 && input1 == 0) // right line detected by right sensor, move right to align
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, BANGBANG_TURNSPEED);
        }
    }
    else
    {
        // middle line controller for "middle line mode"

        if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
        {
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
        }
        else if (input2 == 1) // middle line detected by middle left sensor, move right to align
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, MIDDLE_TURNSPEED);
        }
        else if (input4 == 1) // middle line detected by middle right sensor, move left to align
        {
            analogWrite(motor1f, MIDDLE_TURNSPEED);
            analogWrite(motor2f, 0);
        }
    }
}

/*
 * Function Name:       moveForwardLogicHighBangBang
 * Input:               int right_speed, int left_speed
 * Output:              void
 * Logic:               Logic for moving forward with a higher bangbang speed for 
 *                      certain situations.
 * Example Call:        moveForwardLogicHighBangBang(220, 220)
 */
void moveForwardLogicHighBangBang(int right_speed, int left_speed)
{
    if (input2 == 0 && input3 == 0 && input4 == 0) // bang bang controller for "wall mode"
    {

        if ((input1 == 0 && input5 == 0) || (input1 == 1 && input5 == 1)) // forward if no line detected or both lines detected
        {
            analogWrite(motor1r, 0);
            analogWrite(motor2r, 0);
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
        }
        else if (input1 == 1 && input5 == 0) // left line detected by left sensor, move left to align
        {
            analogWrite(motor1r, 0);
            analogWrite(motor1f, BANGBANG_TURNSPEED + XINCREASEBANGBANG);
            analogWrite(motor2f, 0);
        }
        else if (input5 == 1 && input1 == 0) // right line detected by right sensor, move right to align
        {
            analogWrite(motor2r, 0);
            analogWrite(motor1f, 0);
            analogWrite(motor2f, BANGBANG_TURNSPEED + XINCREASEBANGBANG);
        }
    }
    else
    {
        // middle line controller for "middle line mode"

        if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
        {
            analogWrite(motor1f, left_speed);
            analogWrite(motor2f, right_speed);
        }
        else if (input2 == 1) // middle line detected by middle left sensor, move right to align
        {
            analogWrite(motor1f, 0);
            analogWrite(motor2f, MIDDLE_TURNSPEED);
        }
        else if (input4 == 1) // middle line detected by middle right sensor, move left to align
        {
            analogWrite(motor1f, MIDDLE_TURNSPEED);
            analogWrite(motor2f, 0);
        }
    }
}

/*
 * Function Name:       moveForwardLogicSpecial
 * Input:               int right_speed, int left_speed
 * Output:              void
 * Logic:               Logic for moving forward in certain situations (termination).
 *                      This function removes bangbang (wall) logic from movement.
 * Example Call:        moveForwardLogicSpecial(200, 200)
 */
void moveForwardLogicSpecial(int right_speed, int left_speed)
{
    if (input3 == 1 && (input2 == 0 && input4 == 0)) // move forward if middle line detected only by middle sensor
    {
        analogWrite(motor1f, left_speed);
        analogWrite(motor2f, right_speed);
    }
    else if (input2 == 1) // middle line detected by middle left sensor, move right to align
    {
        analogWrite(motor1f, 0);
        analogWrite(motor2f, MIDDLE_TURNSPEED);
    }
    else if (input4 == 1) // middle line detected by middle right sensor, move left to align
    {
        analogWrite(motor1f, MIDDLE_TURNSPEED);
        analogWrite(motor2f, 0);
    }
}

/* Wifi functions */

/*
 * Function Name:       establishInitialWifiConnection
 * Input:               void
 * Output:              void
 * Logic:               Connects to WiFi and Host, and does not return until it does so.
 * Example Call:        establishInitialWifiConnection()
 */
void establishInitialWifiConnection()
{

    // connect to wifi with ssid and password
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(WIFI_TRY_DELAY);
        Serial.print("Attempting... ");
        Serial.println(ssid);
    }

    Serial.print("WiFi connected with IP: ");
    Serial.println(WiFi.localIP());

    // connect to host computer
    do
    {
        Serial.println(host);
        Serial.println("Connection to host failed... Retrying.");
        delay(CONNECTION_PING_DELAY);
    } while (!client.connect(host, port));

    client.print("ACK_REQ_FROM_ROBOT");
    Serial.println("Connection to host successful");
}

/*
 * Function Name:   listenAndDirectActions
 * Input:           void
 * Output:          void
 * Logic:           Listens on the Wifi connection and communicates to 
 *                  the controlLoop via the queue for any actions. 
 *                  Blocking function.
 * Example Call:    listenAndDirectActions()
 */
void listenAndDirectActions()
{
    while (1)
    {
        // a code that represents the action to be taken, will be sent to the controlLoop
        unsigned int action_code = 0;

        String clientMessage = client.readStringUntil('\n');
        if (clientMessage.length() != 0)
        {
            if (clientMessage == "START")
            {
                // The message after the "START" message is the path
                char message[MESSAGE_QUEUE_SIZE];
                String path = client.readStringUntil('\n');

                path.toCharArray(message, MESSAGE_QUEUE_SIZE);

                if (strlen(message) < 3) {
                    client.print("DNR\n");
                    continue;
                } else {
                    client.print("GOO\n");
                }

                if (xQueueSend(message_queue, message, portMAX_DELAY)) // portMAX_DELAY means block until the message is sent
                {
                    Serial.println("SENT MESSAGE TO MESSAGE_QUEUE");
                }

                action_code = START_ACTIONCODE;
                if (xQueueSend(action_queue, &action_code, portMAX_DELAY))
                {
                    // once this if condition meets, the controlLoop will start!
                    Serial.println("SENT START_ACTION TO ACTION_QUEUE");
                }
            }
            else if (clientMessage == "ISTOP")
            {
                // the host computer might send multiple "ISTOP" messages to stop at an event node.
                // however, only one should be considered, and the rest should be dropped!
                action_code = INTERRUPTSTOP_ACTIONCODE;
                int to_send = 0;
                if (xQueueReceive(reverse_action_queue, &to_send, 0)) // the 0 indicates we don't wait, we just check if there's something in the queue.
                {
                    // if the controlLoop has 'requested' for an ISTOP action, only then do we send it.
                    if (to_send)
                    {
                        if (xQueueSend(action_queue, &action_code, portMAX_DELAY))
                        {
                            // once this if condition meets, the controlLoop will stop, for event node detection.
                            Serial.println("SENT INTERRUPT-STOP TO ACTION_QUEUE");
                        }
                    }
                }
            }
        }

        // send any messages to the host (for debugging purposes)
        char msg_send[MESSAGE_QUEUE_SIZE];
        if (xQueueReceive(send_to_wifi_queue, msg_send, 0))
        {
            client.print(msg_send);
        }
        delay(1); // to prevent the loop from running too fast
    }
}

/* Elementary movement functions */

/*
 * Function Name:   stop
 * Input:           void
 * Output:          void
 * Logic:           Stop all motor movement.
 * Example Call:    stop()
 */
void stop()
{
    analogWrite(motor1f, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1r, 0);
    analogWrite(motor2r, 0);
}

/*
 * Function Name:   turn_right
 * Input:           void
 * Output:          void
 * Logic:           Turn right-wards in-place with `ROTATE_SPEED_RIGHT`
 *                   as the speed value.
 * Example Call:    turn_right()
 */
void turn_right()
{
    analogWrite(motor1r, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1f, ROTATE_SPEED_RIGHT);
    analogWrite(motor2r, ROTATE_SPEED_RIGHT);
}

/*
 * Function Name:   turn_left
 * Input:           void
 * Output:          void
 * Logic:           Turn left-wards in-place with `ROTATE_SPEED_LEFT_L` 
 *                  & `ROTATE_SPEED_LEFT_R` as the speed values.
 * Example Call:    turn_left()
 */
void turn_left()
{
    analogWrite(motor2r, 0);
    analogWrite(motor1f, 0);
    analogWrite(motor1r, ROTATE_SPEED_LEFT_L);
    analogWrite(motor2f, ROTATE_SPEED_LEFT_R);
}

/*
 * Function Name:   turn_right_faster
 * Input:           void
 * Output:          void
 * Logic:           Turn right-wards in-place with `ROTATE_SPEED_UTURN`
 *                  as the speed value. Use for u-turns!
 * Example Call:    turn_right_faster()
 */
void turn_left_slower()
{
    analogWrite(motor2r, 0);
    analogWrite(motor1f, 0);
    analogWrite(motor1r, ROTATE_SPEED_SLOW_LEFT);
    analogWrite(motor2f, ROTATE_SPEED_SLOW_LEFT);
}

/*
 * Function Name:   turn_right_faster
 * Input:           void
 * Output:          void
 * Logic:           Turn right-wards in-place with `ROTATE_SPEED_UTURN`
 *                  as the speed value. Use for u-turns!
 * Example Call:    turn_right_faster()
 */
void turn_right_faster()
{
    analogWrite(motor1r, 0);
    analogWrite(motor2f, 0);
    analogWrite(motor1f, ROTATE_SPEED_UTURN);
    analogWrite(motor2r, ROTATE_SPEED_UTURN);
}

/*
 * Function Name:   turn_left_faster
 * Input:           void
 * Output:          void
 * Logic:           Turn left-wards in-place with `ROTATE_SPEED_UTURN`
 *                  as the speed value. Use for u-turns!
 * Example Call:    turn_left_faster()
 */
void turn_left_faster()
{
    analogWrite(motor2r, 0);
    analogWrite(motor1f, 0);
    analogWrite(motor1r, ROTATE_SPEED_UTURN);
    analogWrite(motor2f, ROTATE_SPEED_UTURN);
}

/*
 * Function Name:   readIRs
 * input:           void
 * Output:          void
 * Logic:           Utility function that polls the IRs
 *                  and updates the global variables.
 * Example Call:    readIRs()
 */
void readIRs()
{
    input1 = digitalRead(IR1);
    input2 = digitalRead(IR2);
    input3 = digitalRead(IR3);
    input4 = digitalRead(IR4);
    input5 = digitalRead(IR5);

    printIRs();
}

/*
 * Function Name:   printIRs
 * input:           void
 * Output:          void
 * Logic:           Utility function that prints the IRs output to serial.
 *                  For debugging.
 * Example Call:    printIRs()
 */
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

/*
 * Function Name:       loop
 * input:               void
 * Output:              void
 * Logic:               This is the internal loop() function that you usually
 *                      use to write your code. This can be left
 *                      empty since we are using threads.
 * Example Call:        Called automatically by the controller.
 */
void loop()
{
    // We can leave this empty since we are running threads!
}

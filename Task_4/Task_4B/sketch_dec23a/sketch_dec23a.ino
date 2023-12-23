int IR1 = 0; // IR sensors pins
int IR2 = 0;
int IR3 = 0;
int IR4 = 0;
int IR5 = 0;

int motor1 = 0; // motor outputs
int motor2 = 0;

int led = 0; // misc
int buzzer = 0;

int speed1 = 255; // motor speeds
int speed2 = 255;

int input1, input2, input3, input4, input5; // inputs of IR sensors, returns LOW when black line

bool mode = 0; // 0 is bang bang, 1 is the middle line

string path = "nnnrlrrnrnln" // n means do nothing, l = left, r = right


int i = 0; // flags
int move = false;
bool atNode = true;
void forward() {
  if (mode == 0) { // bang bang
    if (!input1) { // left IR hits black
      analogWrite(motor1, speed1);
    }
    else if (!input5) { // right IR hits black
    analogWrite(motor2, speed2);
    }
    else {
      analogWrite(motor1, speed1);
      analogWrite(motor2, speed2);
    }
  }
  else { // middle black line
    if (!input3) { // following the line
      analogWrite(motor1, speed1);
      analogWrite(motor2, speed2);
    }
    if (!input4) { // right IR touches black
      analogWrite(motor1, speed1);
    }
    if (!input2) { // left IR touches black
  analogWrite(motor2, speed2);
    }
  }
}
void stop() {
  analogWrite(motor1, 0);
    analogWrite(motor2, 0);
}
void goToNextNode() {
  atNode = false;
  if (!(input2 || input3 || input4)) { // go ahead until middle IRs see black
    forward();
  }
  else {
    stop();
    atNode = true;
  }
}

void setup() {
  // set  modes
 pinMode(IR1, INPUT);
 pinMode(IR2, INPUT);
 pinMode(IR3, INPUT);
 pinMode(IR4, INPUT);
 pinMode(IR5, INPUT);

 pinMode(motor1, OUTPUT);
 pinMode(motor2, OUTPUT);

  pinMode(led, OUTPUT);
 pinMode(buzzer, OUTPUT);
}

void loop() {
  input1 = digitalRead(IR1); // read IR sensors
  input2 = digitalRead(IR2);
  input3 = digitalRead(IR3);
  input4 = digitalRead(IR4);
  input5 = digitalRead(IR5);

  if (!(input2 && input3 && input4)) { // switch to correct mode
  mode = 1;
  }
  else {
    mode = 0;
  }


  if (move) { // start moving
  if (atNode) {
      char command = path[i++];
      if (command == "l") {
        // turn left
      }
      if (command == "r") {
        // turn right
      }
      digitalWrite(led, HIGH);
      digitalWrite(buzzer, HIGH);
      delay(1000);
      digitalWrite(led, LOW);
      digitalWrite(buzzer, LOW);

      analogWrite(motor1, speed1);
      analogWrite(motor2, speed2); // pushing the robot out of the node

      
  }
    goToNextNode();

    if (i == path.length()) {
      move = false; // break out of the loop
    }
  }
}

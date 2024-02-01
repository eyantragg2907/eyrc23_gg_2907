#include <WiFi.h>


#define CONNECTION_PING_DELAY 200
#define WIFI_TRY_DELAY 500

const char *ssid = "pjrWifi";
const char *password = "SimplePass01";
const uint16_t port = 8002;
const char *host = "192.168.187.144";

const int led_red = 2; // misc
const int led_green = 15;

WiFiClient client;

QueueHandle_t queue;

// Dirty loop with delays, YUCK!
void loop1(void *pvParameters)
{
  while (1)
  { 
    digitalWrite(led_green, HIGH);
    digitalWrite(led_red, HIGH);
    delay(500);
    digitalWrite(led_green, LOW);
    delay(500);
    digitalWrite(led_red, LOW);
    Serial.println(millis());

    int element;

    if (xQueueReceive(queue, &element, 0) == pdTRUE) {
      Serial.print(millis());
      Serial.print(" RECV FROM QUEUE: ");
      Serial.println(element);
    }
    delay(500);
  }
}


void loop2(void *pvParameters)
{
    // setting up wifi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(WIFI_TRY_DELAY);
        Serial.println("...");
        Serial.println(ssid);
    }

    Serial.print("WiFi connected with IP: ");
    Serial.println(WiFi.localIP());

    do
    {
        Serial.println("Connection to host failed");
        digitalWrite(led_red, HIGH);
        delay(CONNECTION_PING_DELAY);
    } while (!client.connect(host, port));

    client.print("ACK_REQ_FROM_ROBOT"); // Send an acknowledgement to host(laptop)

    while (1) {
      int i = 1023;
      String clientRead = client.readStringUntil('\n'); // 1 second time out delay
      // Serial.println(clientRead);
      if (clientRead.length() != 0) {
        if (clientRead.startsWith("STOP")) {
          if (xQueueSend(queue, &i, 0) == pdTRUE) {
            Serial.print(millis());
            Serial.println(" SENT TO QUEUE");
          } 
        }
      }
    }
}

void setup()
{
  Serial.begin(115200);
  pinMode(led_red, OUTPUT);
  pinMode(led_green, OUTPUT);

  queue = xQueueCreate( 10, sizeof( int ) );
 
  if(queue == NULL){
    Serial.println("Error creating the queue");
  }

  // Here we create what is normally a timer thing
  // Note the last argument, this is the Core of the ESP it will run on!
  xTaskCreatePinnedToCore(loop1, "loop1", 10000, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(loop2, "loop2", 10000, NULL, 1, NULL, 1);
}

void loop()
{
// And an emtpy loop!
}

// A robust, non-blocking WiFi server for receiving robot control commands.

#include <WiFi.h>

// --- Configuration Constants ---
// IMPORTANT: Fill in your own network details below!
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// If you want the ESP32 to connect to your router instead of creating
// its own network, you will need to change the WiFi.mode() in setup().
// For this project, we are creating an Access Point.

const uint16_t SERVER_PORT = 80; // Port 80 is the standard for HTTP
const long BAUD_RATE = 115200;

// --- Global Variables ---
WiFiServer server(SERVER_PORT);

// Stores the last valid command received from the Python script.
// Initialize to a "stop" command (e.g., 5) for safety.
int robot_command = 5;

void setup() {
  Serial.begin(BAUD_RATE);
  // Wait for Serial to be ready, but don't block forever.
  while (!Serial && millis() < 5000) {
    delay(10);
  }

  Serial.println("\nConfiguring Access Point...");
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_SSID, WIFI_PASSWORD);

  IPAddress ip = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(ip);

  server.begin();
  Serial.print("HTTP Server started on port ");
  Serial.println(SERVER_PORT);
}

void loop() {
  // This is the main control loop. It should run as fast as possible.
  // We handle incoming clients and then immediately control the robot.

  handle_new_client();

  // The robot's movement is handled on every loop based on the last command.
  handle_robot_movement(robot_command);

  // Add a small delay to prevent overwhelming the ESP32 in a tight loop,
  // especially if there is no other blocking code.
  delay(1);
}

/**
 * @brief Checks for and handles incoming network clients.
 * This function is non-blocking. It processes the request and returns,
 * allowing the main loop to continue running smoothly.
 */
void handle_new_client() {
  WiFiClient client = server.available(); // Check for a new client

  if (client) {
    Serial.println("New client connected.");
    // Use a timeout to prevent the server from hanging on a bad request.
    unsigned long startTime = millis();
    while (client.connected() && (millis() - startTime < 1000)) { // 1 sec timeout
      if (client.available()) {
        // Read the first line of the HTTP request
        String requestLine = client.readStringUntil('\r');
        Serial.print("Request: ");
        Serial.println(requestLine);

        // --- CRITICAL BUG FIX: Parse the command from the URL ---
        int new_command = parse_command_from_request(requestLine);

        if (new_command != -1) {
          // Only update and print if the command has changed.
          if (new_command != robot_command) {
            robot_command = new_command;
            Serial.print("--> New command received: ");
            Serial.println(robot_command);
          }
        }

        // Send a minimal HTTP 200 OK response
        client.println("HTTP/1.1 200 OK");
        client.println("Content-Type: text/plain");
        client.println(); // End of headers
        client.println("Command Received");
        break; // Exit the while loop once the request is handled
      }
    }
    // Give the client time to receive the data, then close the connection.
    delay(1);
    client.stop();
    Serial.println("Client disconnected.");
  }
}

/**
 * @brief Parses the integer command from an HTTP GET request line.
 * Example: "GET /?command=6 HTTP/1.1" -> returns 6
 * @param line The first line of the HTTP request.
 * @return The integer command, or -1 if not found.
 */
int parse_command_from_request(String line) {
  // Find the position of "?command="
  int command_index = line.indexOf("?command=");
  if (command_index != -1) {
    // The actual number starts after the "?command=" string (length 10)
    int value_start_index = command_index + 9;
    // Extract the substring from there to the end and convert to integer
    return line.substring(value_start_index).toInt();
  }
  return -1; // Return -1 to indicate an error or no command found
}


/**
 * @brief Controls the robot's motors based on the command.
 * This is where you will add your motor control logic (e.g., using PWM).
 * @param command The integer command for the desired action.
 */
void handle_robot_movement(int command) {
  // Placeholder for your motor control logic (e.g., using a motor driver)
  // 0=Forward, 1=Left, 2=Right, 3=Backward, 5=Stop, 6=Start, 9=Victory
  switch (command) {
    case 0:
      // go_forward();
      break;
    case 1:
      // turn_left();
      break;
    case 2:
      // turn_right();
      break;
    case 3:
      // go_backward();
      break;
    case 5:
      // stop_motors();
      break;
    // Cases 6 (Start) and 9 (Victory) might not need motor action,
    // but could trigger a light or a sound.
    case 6:
    case 9:
      // stop_motors(); // Good practice to ensure it's stopped
      break;
  }
}
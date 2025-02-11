# Cat Recognition Smart Feeder

## Overview
The **Cat Recognition Smart Feeder** project aims to better control access to food in multi-cat homes. The system uses a TensorFlow camera-based recognition system to feed each cat a set amount of food per day and has an alarm that sounds when one cat is stealing the other's food.

## Team Member
- **Justin Mikolajcik**  
  Contact: [mik19003@byui.edu](mailto:mik19003@byui.edu)

## Stakeholders
- **Justin Mikolajcik** (Owner, Developer)
- **Pet Owners** (End Users)
- **Potential Future Employers** (For showcasing the project on my resume)

## Project Purpose
This project solves a real-world problem by preventing the overweight cat from eating food meant for my underweight kitten. The system will automatically control access to food bowls based on which cat approaches using image recognition.

## Background & Prior Knowledge
I have intermediate experience with Raspberry Pi, Arduino, and Python, but this is my first project involving computer vision. The project builds on:
- **Previous Work**: RFID-based cat feeders exist but require collars. This project uses camera-based recognition to avoid the need for collars.
- **My Experience**: Knowledge of embedded systems, hardware setups, and Python programming.

## Project Description
The **Cat Recognition Smart Feeder** will utilize a Raspberry Pi Camera Module 3 to recognize which cat approaches the food bowl based on their appearance (color or a custom machine learning model). When the correct cat is detected, food access is granted. If the larger cat approaches the smaller cat’s bowl, an alert will be triggered.

### Key Features
- **Camera-based Recognition**: Uses visible light to identify the cat approaching the bowl.
- **Power Efficiency**: A proximity sensor triggers the camera to activate only when a cat is near.
- **Non-invasive**: No collars or RFID tags required; visual recognition is used to distinguish between the cats.

### Audience
- **Primary Audience**: Pet owners with multiple cats.
- **Geographic Reach**: Applicable to pet owners globally.

### Completion Criteria
The project is considered complete when the system reliably differentiates between the two cats and controls access to the food accordingly.

## Significance
This project demonstrates skills in computer vision, hardware integration, and software development, providing a valuable portfolio piece for potential employers. It showcases my ability to integrate technologies like Raspberry Pi, sensors, and machine learning to solve real-world problems.

## New Computer Science Concepts
This project will require learning and implementing:
- **Computer Vision**: Using OpenCV or a deep learning model for image recognition.
- **Machine Learning**: Potentially training a custom model to distinguish between the cats.
- **Hardware Integration**: Combining sensors, relays, and motors with Raspberry Pi for controlling the feeding system.

## Personal Interest
I find this project exciting because it addresses a problem I face daily, combines my interests in machine learning and hardware, and offers practical real-world solutions for multi-pet households.

## Project Milestones and Schedule

| Milestone                      | Task                                                    | Hours | Deadline          |
|---------------------------------|---------------------------------------------------------|-------|-------------------|
| **Milestone 1: Project Planning**   | Finalize idea, proposal, and research components           | 12    | Week 2 (Proposal)  |
| **Milestone 2: Hardware Setup**     | Set up Raspberry Pi, camera, proximity sensor, test hardware | 16    | Week 4             |
| **Milestone 3: Software Development**| Develop camera triggering and detection code using OpenCV   | 28    | Week 8             |
| **Milestone 4: Cat Recognition Model** | Implement and train recognition system (color-based or ML) | 30    | Week 10            |
| **Milestone 5: Integration**        | Combine recognition with feeding mechanism and control    | 24    | Week 12            |
| **Milestone 6: Testing**            | Full system test and adjustments for real-time recognition | 20    | Week 14            |

**Total Hours**: 130

## Resources

### Hardware
- **Raspberry Pi 5** (owned)
- **Raspberry Pi Camera Module 3** (~$30)
- **Proximity sensor** (~$2)
- **Servo motor** (~$3)
- **Night light or LED strip** (~$10)

### Software
- **TensorFlow** (free)
- **OpenCV** (free)
- **Python** (free)
- **Raspberry Pi OS** (free)

**Total Estimated Cost**: ~$45

## Dependencies
- **Languages**: Python (Raspberry Pi OS)
- **IDE**: VSCode or Thonny
- **Hardware**: Raspberry Pi 5, Camera Module 3, sensors
- **Libraries**: OpenCV, possibly TensorFlow (for ML)
- **Deployment**: Developed and tested on Raspberry Pi

## Risks
- **Learning Curve**: Gaining proficiency in OpenCV and basic computer vision.
- **Recognition Accuracy**: Ensuring reliable differentiation between the two cats.
- **Lighting Conditions**: Maintaining detection accuracy in low-light environments.

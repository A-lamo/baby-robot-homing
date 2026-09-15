# Notes and Instructions on Hardware Implementation
This doc aimed to provide a guide on how to assemble, test, and run a modular baby robot. The python scripts mentioned are all from this repository. 

## 1. RPi Setting Up
This steps are recommended before assembling the robot, to verify whether the head of the robot are functioning well, and prepare for the later testing during assembling.
* Note that start this step and the following testing steps, after fully assembled the robot is also fine, but making adjustments after fully assembling will require disassemble parts back and forth, which is annoying and waste of time. Disassemble and assemble over and over again might also damage the 3d-printed parts. 

Required: A robot head with a sd-card. If possible, chose a sd-card with larger memory (32GB is preferred over 8GB).
> Always keep in mind:
> 1. Charging batteries is only permissible when you keep an eye on them. 
> 2. Don't left battery in the robot. The battery keeps discharging to the point that it starts messing with the actual robot until the head breaks. So please make sure to remove the batteries.

1. Flash the microSD card with Raspberry Pi OS 
	1. Download Raspberry Pi Imager, a quick tool to install Raspberry Pi OS and other operating systems to a microSD card.
		1. Device: `Raspberry 4`
		2. OS: `Raspberry pi OS (64 bits)`
		3. Set hostname
		4. Set username and password
			* Note that the hostname, username and password names will be regularly used, so in practice, make them as short as possible. E.g. let all of them be `pi`
		5. Keyboard layout: *us* ( It can always be changed using the raspi-config tool on the RPi)
		6. Wifi: connect to **Thymionet**
		7. Enable SSH, use password authentication
		8. Enable Raspberry Pi Connection.
2. Prepare SSH:
	1. Connect the RPi and your own laptop to **Thymionet**
	2. Open the terminal (can use `ctrl+alt+t`), find the ip address by `ipconfig`.
	3. Ssh to it in terminal: `ssh <username>@<ip address>` (if not)
3. Install necessary packages: 
	1. Install python.
		* It might be helpful to deploy a lightweighted package management tool such as uv and pyenv. 
	2. Install numpy&opency: `pip install numpy,opencv-python`
	3. (If required) Install cpu-only torch: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
	4. If the robot has a camera, verify [[picamera2]] installation ([documentation](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-2-picamera2-manual.pdf)) in the RPi:
		1. Check by `sudo apt install -y python3-picamera2`
		2. To update *picamera2*, use `sudo apt update` followed by `sudo apt full-upgrade`
4. Install CI Group's *robot-hat* library ([git repo](https://github.com/sunfounder/robot-hat)) in the RPi:
	1. Download the repo: `sudo apt install -y git`, `cd ~`, `git clone git@github.com:ci-group/robohat.git`
	2. Update packages and fix the GPIO dependency: 
		1. ```bash
		   sudo apt update
		   sudo apt upgrade -y
		   sudo apt remove -y python3-rpi.gpio
		   sudo apt install -y python3-rpi-lgpio i2c-tools
		   ```
	   3. Then: `sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.backup`, `sudo cp ~/robohat/setup_files/config.txt /boot/firmware/config.txt`, then `sudo reboot`
	   4. Put the files where the library expects them: 
		   ```bash
		   mkdir -p ~/bin ~/robohat
			cp ~/robohat/bin/robo ~/robohat/bin/servo ~/robohat/bin/buzz_random ~/bin/
			cp -r ~/robohat/robohatlib ~/robohat/
			cp -r ~/robohat/testlib   ~/robohat/
			cp ~/robohat/Test.py ~/robohat_repo/SerTest.py ~/robohat/
		   ```

## 2. Recommended Building Sequence 
The following steps aim to assemble a robot as close as the robot at rest in simulator. 

Building blocks required: A head, servo motors, brick modules, hinge parts.
1. Before starting assembling, **connect only one servo** firmly to the core, then try to run `test_servos_sequential.py`, to verify whether the general setting is correct (or alternative tests described below). The desired effect is a sweep-over from $0 \degree$ to $180 \degree$.
	* Note that during my implementation, ARIEL's default hinges are $180 \degree$ hinges. 
	So make sure to avoid using those $270 \degree$ hinges, which might cause some problem.
	* The servo type that I used and managed to have a successful demo is DSS-M15 with a specific wire color. 
	Other servo types might also work, but might also require different testing script setting. ![servo image](assets/servo.png)
	* An alternative is to run `robohat` library's built-in `robo` and `servo` tests to achieve similar effects. 
2. **Connect required number of servos** to the servo boards inside the head. Then run `test_servos_sequential.py` again to test whether all servos are functioning well.
	* If a servo is week (i.e. can only sweep over a range way smaller than $[0 \degree, 180 \degree]$) or completely not working, make sure to replace it. 
	* If the servo cannot be detected by `robohat` program or the test script: first try to plug the cable from the other way around, then try another port. From experience, there are two types of servos in the lab that required different direction of connection. 
3. **Assemble and connect the hinges**: assemble and connect the hinges to the servo boards. The following sub-steps and principles should be followed:
	1. Decide which side in the head is dedicated to be the front side of the robot (better to left a tape mark). Note that the side with servo board should be the up side. 
	3. Assemble the hinge units with **a test on servos's neutral position**: 
		1. Loosely assemble the hinges (i.e. assemble the hinge parts and the servo together, but do not screw on yet). Connect the hinges to servo boards. 
			* **A principle regarding the connection of servos**: connecting the servos to *different* servo boards (i.e. assemblies, there are 2 inside the head), to avoid possible power/current-load problem. 
				* E.g. Puts forelimb/left-front group on board 1 and spine/hind group on board 2
				* If the servos are all connecting to the one assembly board, it might be overloaded during runtime, thus not be able to deliver the requested simultaneous motion cleanly.
		2. Run the build-in `servo` test in the `robohat` lib, then enter 5 to move all servos to their neutral position (i.e. to 90 degrees position from its 0-180 degrees range).
			* Make sure to avoid rotate the hinge unit by hand when a program is running.
		3. Check whether the natural position of the servo can make the hinge unit stay relatively "straight". If not, either adjust the hinge parts or plug to other ports. From experience, the actual neutral position of each servo changes when you plug it to another port.
			* For example, the first image is more "straight" thus more preferred than the second image (ignore the screws). If a hinge appears like the second image, either adjust the hinge parts or plug to other ports. It is normal that a hinge can not be perfectly straight, but the principle is that letting them be as straight as possible. Step 5 will adjust them further.![hinge prefer](assets/hinge_prefer.png) ![hinge not prefer](assets/hinge_not_prefer.png)
			* The aim is: This steps is aimed for making sure that when the robot is at rest, the position of the robot is as close as the rest posture (i.e. the positions of each joints before robot start moving) in simulator. E.g. the first natural posture is preferable than the second natural posture. ![robot neutral](assets/robot_neutral.png)![robot less neutral](assets/robot_less_neutral.png) 
		4. Use screws to fully fix each hinge, while keep the servo test 5 running. Mark on each servo the board port it is expected to connect with (e.g. put a paper tape or paper tag on the wire/servo body to note down "right board, port 0"). Then unplug those wires.
			* This is because when you plug the wire to another port, the natural position might change, so it will require redo this step and step 5 for this hinge.
4. **Fully assemble the robot**: Align carefully to the robot in simulator, especially be careful on the direction of hinges. 
	* It might be helpful in this stage to visualize the robot in simulator. Customize and run `helper/visualizer_sim/visualize_robot.py` in Ariel can be helpful for this work.
	* **A principle regarding the hinges direction**: To align the actual robot with ARIEL's prebuilt or customized robot with `src/ariel/body_phenotypes/robogen_lite/prebuilt_robots/robot_builder.html`, the direction of hinges should always following a specific "convention", where the stators (the part that fix the servo) are always connecting to the previous parts relative to the core; and the motor (the part that the motor lifts) are always connecting next parts. In deployment, it means that the servo wires are always pointing out of the core. 
		* For reference, the image below shows one hinges, where the upper left part is the motor which connect to the module, and bottom part is the stator that connect to the core. Note that the screws placement in this image are not the standard setting. ![hinge](assets/hinge.png)
		* A robot in sim and in real are shown in the following images as an example. The darker parts in sim are motors, which in real world should be the part that is wrapping the another. ![baby robot sim](assets/baby_robot_sim.png) ![baby robot real](assets/baby_robot_real.png)
			* In this assembling, the up-down moving hinges in the forelimbs are placing the small black connectors to the front, while in the hind limbs, its the opposite. This is a deliberate choice, since if placing in the other way around, when the movement of hind limbs are aggressive, the end brick modules might collide with the forelimbs and the wires and the extended part of the black connector might pull each other. So to ensure safety, this specific way is applied.  
5. **Determine manually the neutral position**: 
	> Apart from the efforts in 3, the hinges might still be slightly from the neutral position. A mitigation strategy here is to let the robot start from the designated neutral position instead of the default natural neutral position (i.e. $90 \degree$). So the actual commanded is 90 + the raw commanded degrees from the robot controller, then clamping into $[0 \degree, 180 \degree]$. The actual implementation is in `hardware/baby_hardware.py`. 
	1. Run command `robo`, then command `set servo angle <servo nr> <angle>` repeatedly for every servo to find an angle that can make that servo go to visually neutral position (i.e. straight).
		* E.g. `set servo angle 0 85`, `set servo angle 0 95`.
		* Note that the two servo boards' port number start from 0 to 31 as labeled in the chip. To determine the servo numbers, either try manually or run `test_servos_sequential.py`.
	2. Edit `hardware/baby_hardware.py` in `DEFAULT_SERVO_MAPPINGS -> neutral_deg`.
		* It will also be helpful to also modify the third column of the DEFAULT_SERVO_MAPPINGS, to map between the channel in the simulated robot in the second column to the servo channel of real robot. 
6. **Determine the signs**: 
	> Depending on the hinges' actual direction, the output movement of the servos might be opposite of what in the simulated robot. E.g. in simulator, a joint that is commanded to move up, but in real robot, that joint is actually moving down. For those joint, we add a negative sign to the commanded angle, which can be checked with the following guide.
	1. Run `helper/calibration/record_hinge_sweep.py` to record a video that let each joint sweep over its minimum to maximum allowed angles (in Mujoco simulator, that means [-90 rad, +90 rad]). 
	2. Run `test_servos_sequential.py` to observe the actual sweeping of each joints. Record the joint number that is different from what is in the simulator. Then edit `DEFAULT_SERVO_MAPPINGS` in `hardware/baby_hardware.py` -> `sign`

## 3. Runtime Operations
* **How to turn on the robot**: hold the small yellow button until the red light start constantly flickering, then immediately release it. Redo if not succeed. 
* **How to shutdown**: `sudo shutdown now`
* To transfer files between the RPi and and the local device, there are two possible ways: 
	1. Via SSH: easier and safer way
	2. Via git: more convenient, but the repo might be corrupted if the robot shutdown while the files are transferring, which is a bit tricky to fix.
* If want to transfer files via git, the following initialization steps should help: 
	1. Clone the project's git repo (via SSH).If SSH certification issue in git clone occurs, try the following debug steps:
		 1. Date and time:
			 1. Check if the time and data is wrong: `date` 
			 2. Trigger a time sync: `sudo timedatectl set-ntp true`
		 2. Or:`sudo apt install -y htpdate fake-hwclock`, `sudo htpdate -s www.google.com`
		 3. Update certificate: `sudo apt update` and `sudo apt install --only-upgrade ca-certificates`
		 4. Authenticate pi with GitHub:
			 1. Generate an SSH Key on your Raspberry Pi: `ssh-keygen -t ed25519 -C "<git username>"`i
			 2. Copy the SSH key: `cat ~/.ssh/id_ed25519.pub`
		 5. Add the key to your GitHub account
			1. Go to GitHub and log in.
			2. In the top-right corner, click your profile picture and go to Settings.
			3. In the left sidebar, click SSH and GPG keys.
			4. Click the green New SSH key button.
			5. Give it a Title (e.g., "Raspberry Pi").
			6. Paste your copied key into the Key field.
			7. Click Add SSH key.
		6. Clone the private repo via SSH. But to avoid future memory issues, do not download Ariel in the Pi and run `uv sync`.
	2. Allow git commit and push
		1. Set: `git config --global user.email "your_email@example.com"`
		2. And: `git config --global user.name "<username>"`
- **Check remaining storage in the RPi**: `du -h --max-depth=1 ~ 2>/dev/null | sort -h | tail -20`, 
- **Delete a directory**:`rm -r <path>`
- **Copy file from Local to SSH**: ` scp <path to local file> <username>@<ip address>:/home/<username>`
	- A folder: ` scp -r <path to folder> <username>@<ip address>:/home/robohat`
- **Copy file from SSH to local**: `scp PATH <local username>@<local ip address>:<local path>`

## 4. Troubleshooting
1. When can successfully boost the OS by power cable but not battery: check the connection of the yellow cable to both the Power Manager board and RPi. 
2. If the servos can be detected by `servo` test but cannot be manipulated: check the servo number 16-31 instead of 0-15.
3. `ssh: connect to host <ip addr> port <nr>: Host is down`, or Operation timed out.
	1. Try connect to the monitor. If the OS cannot be successfully launched (a rainbow color square size appearing and disappearing repeatedly), try to charge the battery then try again. 
		1. I guess it is because the power supply of the battery is not stable/strong enough to supply the RPi. So replace the battery can help.
5. If ssh connection is not successful, first check if the laptop is connected to Thymionet before touching the robot.

## 5. Remaining Issues and Suggestions for Hardware Design
1. Can add an extra connection place in the hinge's rotor, two connections are not strong enough for aggressive movement in long arms/legs
2. Adding another two holes for screws at the robot's bottom lid. If only use 2, the robot is not stable enough. Also, adding screws will lift the robot's head a little bit, then the robot's limb cannot touch the ground anymore (like in the simulator). This might add an extra layer of errors. 
3. Friction is a problem when running the robot. Wrap the modules with tape might help. 

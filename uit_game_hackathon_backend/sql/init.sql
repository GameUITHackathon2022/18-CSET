CREATE DATABASE IF NOT EXISTS  uit_game_hackathon CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-----------------------------------------------------
-----------------------------------------------------
USE uit_game_hackathon
---------------------------------------------
-- USER
CREATE TABLE IF NOT EXISTS  User(
	_id	   INT NOT NULL AUTO_INCREMENT,	
	username VARCHAR(255),
	email      VARCHAR(255),
	score int,
	create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (_id),
	user_password VARCHAR(255),
	CONSTRAINT user_unique UNIQUE (username,user_password)
);
---------------------------------------------
-- Camera
CREATE TABLE IF NOT EXISTS Camera(
	_id	   INT NOT NULL AUTO_INCREMENT,	
	cameraname VARCHAR(255),
	ip VARCHAR(255),
	username VARCHAR(255),
	camera_password VARCHAR(255),
	protocol VARCHAR(255),
	location_desc VARCHAR(100),
	create_at DATETIME DEFAULT CURRENT_TIMESTAMP,


	PRIMARY KEY (_id)
);
---------------------------------------------
-- Detector
CREATE TABLE IF NOT EXISTS Mission(
	_id INT NOT NULL AUTO_INCREMENT,
	camera_id int,
	img_url VARCHAR(40),
	mission_score int,
	location_desc VARCHAR(100),
	is_done tinyint,
	is_done_model tinyint,
	mission_time DATETIME DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (_id)
);

CREATE TABLE IF NOT EXISTS Joiner(
	userid int not null,
	mission_id int not null,
	PRIMARY KEY (userid,mission_id)
);

CREATE TABLE IF NOT EXISTS Gift(
	_id INT NOT NULL AUTO_INCREMENT,
	code VARCHAR(200),
	type_gift VARCHAR(200),
	avail tinyint,
	price int,

	PRIMARY KEY (_id)
);


ALTER TABLE Mission
ADD FOREIGN KEY (camera_id) REFERENCES Camera(_id);
ALTER TABLE Joiner
ADD FOREIGN KEY (userid) REFERENCES User(_id);
ALTER TABLE Joiner
ADD FOREIGN KEY (mission_id) REFERENCES Mission(_id);


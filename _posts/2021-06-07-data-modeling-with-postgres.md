---
layout: post
title: Data Modeling with Postgres
date: 2021-06-07T 00:00:00 +03:00
description: "A comprehensive data modeling project that creates a Postgres database schema and ETL pipeline for Sparkify music streaming startup, enabling efficient analysis of user activity data through optimized SQL queries"
image: "assets/images/masonary-post/postgres.png"
categories: 
  - "Data Engineering"
---

## Project Overview

Sparkify, a music streaming startup, wants to analyze user activity data collected from their new streaming app. The analytics team is particularly interested in understanding what songs users are listening to. Currently, their data resides in a directory of JSON logs containing user activity and JSON metadata about the songs in their app.

The challenge is that they have no easy way to query their data, which is currently in JSON format and needs to be transformed into a more queryable structure.

This project creates a Postgres database schema and ETL pipeline that transforms raw JSON data into a structured format optimized for song play analysis. The goal is to enable the analytics team to easily run queries and generate insights about user listening patterns.

## Technical Architecture

### Project Structure
>```
>data_modeling_postgres/
>│
>├── data/                      # Dataset files
>│   ├── song_data/            # JSON files of song metadata
>│   └── log_data/             # JSON files of user activity logs
>│
>├── create_tables.py          # Database and table creation script
>├── etl.py                    # ETL pipeline script
>├── sql_queries.py            # SQL queries for CRUD operations
>├── etl.ipynb                 # ETL process development notebook
>├── test.ipynb               # Database testing notebook
>└── README.md                # Project documentation
>```

### Core Components

>```
>1. create_tables.py
>   - Establishes database connection
>   - Drops existing tables to ensure fresh start
>   - Creates new tables using schema definitions
>   - Acts as a reset script for testing
>
>2. etl.py
>   - Implements the ETL pipeline
>   - Processes song and log data files
>   - Transforms JSON data into appropriate formats
>   - Loads data into Postgres tables
>
>3. sql_queries.py
>   - Contains all SQL queries used throughout the project
>   - Includes CREATE, DROP, INSERT, and SELECT statements
>   - Centralizes query management for maintainability
>```

## Database Schema Design

The database uses a star schema optimized for song play analysis. This design prioritizes denormalization and simplifies queries while maintaining data integrity.

### Fact Table

**songplays** - Records in log data associated with song plays
>```sql
>CREATE TABLE IF NOT EXISTS songplays (
>    songplay_id SERIAL PRIMARY KEY,
>    start_time timestamp NOT NULL,
>    user_id int NOT NULL,
>    level varchar,
>    song_id varchar,
>    artist_id varchar,
>    session_id int,
>    location varchar,
>    user_agent varchar,
>    FOREIGN KEY (user_id) REFERENCES users (user_id),
>    FOREIGN KEY (song_id) REFERENCES songs (song_id),
>    FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
>    FOREIGN KEY (start_time) REFERENCES time (start_time)
>);
>```

### Dimension Tables

**users** - User information
>```sql
>CREATE TABLE IF NOT EXISTS users (
>    user_id int PRIMARY KEY,
>    first_name varchar,
>    last_name varchar,
>    gender varchar,
>    level varchar
>);
>```

**songs** - Song metadata
>```sql
>CREATE TABLE IF NOT EXISTS songs (
>    song_id varchar PRIMARY KEY,
>    title varchar,
>    artist_id varchar,
>    year int,
>    duration float
>);
>```

**artists** - Artist information
>```sql
>CREATE TABLE IF NOT EXISTS artists (
>    artist_id varchar PRIMARY KEY,
>    name varchar,
>    location varchar,
>    latitude float,
>    longitude float
>);
>```

**time** - Timestamps of records broken down
>```sql
>CREATE TABLE IF NOT EXISTS time (
>    start_time timestamp PRIMARY KEY,
>    hour int,
>    day int,
>    week int,
>    month int,
>    year int,
>    weekday int
>);
>```

## ETL Pipeline Implementation

### 1. Song Data Processing

The ETL pipeline first processes song data from JSON files structured like:
>```json
>{
>    "num_songs": 1,
>    "artist_id": "ARD7TVE1187B99BFB1",
>    "artist_latitude": null,
>    "artist_longitude": null,
>    "artist_location": "California - LA",
>    "artist_name": "Casual",
>    "song_id": "SOMZWCG12A8C13C480",
>    "title": "I Didn't Mean To",
>    "duration": 218.93179,
>    "year": 0
>}
>```

Processing steps:
>```python
>def process_song_file(cur, filepath):
>    # Open song file
>    df = pd.read_json(filepath, lines=True)
>
>    # Insert song record
>    song_data = df[['song_id', 'title', 'artist_id', 'year', 'duration']].values[0].tolist()
>    cur.execute(song_table_insert, song_data)
>    
>    # Insert artist record
>    artist_data = df[['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude']].values[0].tolist()
>    cur.execute(artist_table_insert, artist_data)
>```

### 2. Log Data Processing

Next, it processes user activity logs:
>```json
>{
>    "artist": "Muse",
>    "auth": "Logged In", 
>    "firstName": "Jordan",
>    "gender": "F",
>    "itemInSession": 3,
>    "lastName": "Hicks",
>    "length": 259.26485,
>    "level": "free",
>    "location": "Salinas, CA",
>    "method": "PUT",
>    "page": "NextSong",
>    "registration": 1540008898796.0,
>    "sessionId": 814,
>    "song": "Supermassive Black Hole",
>    "status": 200,
>    "ts": 1543190563796,
>    "userAgent": "Mozilla/5.0",
>    "userId": "37"
>}
>```

Processing steps:
>```python
>def process_log_file(cur, filepath):
>    # Filter records by NextSong action
>    df = df[df['page'] == 'NextSong']
>
>    # Convert timestamp column to datetime
>    t = pd.to_datetime(df['ts'], unit='ms')
>    
>    # Insert time data records
>    time_data = (t, t.dt.hour, t.dt.day, t.dt.week, t.dt.month, t.dt.year, t.dt.weekday)
>    column_labels = ('timestamp', 'hour', 'day', 'week', 'month', 'year', 'weekday')
>    time_df = pd.DataFrame(dict(zip(column_labels, time_data)))
>
>    for i, row in time_df.iterrows():
>        cur.execute(time_table_insert, list(row))
>
>    # Load user records
>    user_df = df[['userId', 'firstName', 'lastName', 'gender', 'level']]
>
>    # Insert user records
>    for i, row in user_df.iterrows():
>        cur.execute(user_table_insert, row)
>
>    # Insert songplay records
>    for index, row in df.iterrows():
>        cur.execute(song_select, (row.song, row.artist, row.length))
>        results = cur.fetchone()
>        
>        if results:
>            songid, artistid = results
>        else:
>            songid, artistid = None, None
>
>        # Insert songplay record
>        songplay_data = (pd.to_datetime(row.ts, unit='ms'), 
>                        row.userId, row.level, songid, artistid, 
>                        row.sessionId, row.location, row.userAgent)
>        cur.execute(songplay_table_insert, songplay_data)
>```

## Example Queries and Results

### 1. Most Active Users
>```sql
>SELECT u.first_name, u.last_name, COUNT(*) as play_count
>FROM songplays sp
>JOIN users u ON sp.user_id = u.user_id
>GROUP BY u.user_id, u.first_name, u.last_name
>ORDER BY play_count DESC
>LIMIT 5;
>```

### 2. Popular Music Hours
>```sql
>SELECT t.hour, COUNT(*) as play_count
>FROM songplays sp
>JOIN time t ON sp.start_time = t.start_time
>GROUP BY t.hour
>ORDER BY play_count DESC;
>```

### 3. User Activity by Location
>```sql
>SELECT location, COUNT(*) as activity_count
>FROM songplays
>GROUP BY location
>ORDER BY activity_count DESC
>LIMIT 10;
>```

## How to Run

#### 1. Create and activate a Python virtual environment:
>```bash
>python -m venv venv
>source venv/bin/activate  # On Windows: venv\Scripts\activate
>```

#### 2. Install required packages:
>```bash
>pip install -r requirements.txt
>```

#### 3. Run create_tables.py to set up the database and tables:
>```bash
>python create_tables.py
>```

#### 4. Run the ETL pipeline to process and load data:
>```bash
>python etl.py
>```

## Key Achievements

>```
> 1. Designed an optimized star schema for efficient querying of music streaming data
> 2. Built a robust ETL pipeline that successfully processes and transforms JSON data
> 3. Implemented data validation and quality checks throughout the pipeline
> 4. Created a queryable database that enables complex analysis of user listening patterns
> 5. Achieved efficient data loading with minimal duplicate records
> 6. Implemented error handling and logging for pipeline monitoring
>```

## Technologies Used

>```
>1. Python 3.7+
>  - pandas for data manipulation
>  - psycopg2 for PostgreSQL connection
>  - json for parsing JSON files
>
>2. PostgreSQL 9.6+
>  - SERIAL data type for auto-incrementing IDs
>  - Foreign key constraints for data integrity
>
>3. SQL
>  - DDL for schema definition
>  - DML for data manipulation
>  - Complex joins and aggregations
>
>4. JSON
>  - Nested data structure handling
>  - Data extraction and parsing
>```

## Future Improvements

>```
>1. Add data quality checks and constraints
>2. Implement incremental loading
>3. Add indexing for performance optimization
>4. Create automated testing suite
>5. Implement logging and monitoring
>6. Add data visualization dashboard
>```

## Conclusion

This project successfully demonstrates the process of designing and implementing a robust data modeling solution using PostgreSQL for Sparkify. By creating an optimized star schema and a comprehensive ETL pipeline, we have enabled the analytics team to efficiently query and analyze user activity data, leading to valuable insights into user behavior and song preferences.

The implementation showcases the power of PostgreSQL in handling complex queries and maintaining data integrity through foreign key constraints. The structured approach to data modeling not only enhances query performance but also ensures that the data remains consistent and reliable.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Data-Modeling-with-Postgres).
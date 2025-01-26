---
layout: post
title: Building a Data Lake with Spark
date: 2021-07-07T 14:30:00 +03:00
description: "Creating an ETL pipeline to transform and load data into a data lake using Apache Spark for a music streaming startup."
image: "assets/images/masonary-post/datalake.png"
categories: 
  - "Data Engineering"
---

## [Building a Data Lake with Spark](https://github.com/shrikantnaidu/Data-Lake-with-Spark)

---

### Why We're Here

As Sparkify, a music streaming startup, continues to grow its user base and song database, the need to transition from a traditional data warehouse to a more scalable data lake becomes essential. The existing data resides in Amazon S3, consisting of JSON logs on user activity and metadata on songs. The objective of this project is to build an ETL (Extract, Transform, Load) pipeline that processes this data using Apache Spark and loads it back into S3 as dimensional tables.

### Project Overview

The project involves the following key steps:


- a. **Extract** data from S3.
- b. **Transform** the data using Spark to create a set of dimensional tables.
- c. **Load** the transformed data back into S3 in a format suitable for analytics.


### Database Schema

#### Fact Table

**songplays**: Records in event data associated with song plays (i.e., records with page `NextSong`).
>```
>songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
>```

#### Dimension Tables

**users**: Users in the app.
>```
>user_id, first_name, last_name, gender, level
>```

**songs**: Songs in the music database.
>```
>song_id, title, artist_id, year, duration
>```

**artists**: Artists in the music database.
>```
>artist_id, name, location, latitude, longitude
>```

**time**: Timestamps of records in songplays broken down into specific units.
>```
>start_time, hour, day, week, month, year, weekday
>```

### Implementation Details

The implementation consists of the following files:

- a. **etl.py**: This script reads and processes files from `song_data` and `log_data`, transforming them into the defined tables and loading them into S3.
- b. **dl.cfg**: Contains the credentials to access cloud resources.
- c. **README.md**: Documentation of the process, providing execution information for the project.

### Process Overview

a. **Read the Data from S3**:
   - i. **Song data**: The song data is stored in JSON format in the S3 bucket at `s3://udacity-dend/song_data`. This dataset contains metadata about songs, including artist information, song titles, and durations.
   - ii. **Log data**: The log data is also stored in JSON format at `s3://udacity-dend/log_data`. This dataset contains user activity logs, including song plays, user information, and timestamps.

b. **Process Data Using Spark**:
   - i. The script uses Apache Spark to read the JSON files from S3. Spark's distributed computing capabilities allow for efficient processing of large datasets.
   - ii. The data is transformed into five different tables:
     - • **Fact Table**: 
       - • `songplays` - records in log data associated with song plays. This table captures the relationship between users and the songs they listen to.
     - • **Dimension Tables**: 
       - • `users`: Contains user information, allowing for analysis of user behavior.
       - • `songs`: Contains song metadata, enabling insights into song popularity.
       - • `artists`: Contains artist information, facilitating analysis of artist performance.
       - • `time`: Contains time-related data, allowing for time-based analysis of song plays.

c. **Load the Data Back to S3**:
   - i. The transformed data is written back to S3 in Parquet format, which is optimized for analytics. Parquet files are columnar storage files that provide efficient data compression and encoding schemes, making them ideal for big data processing.
   - ii. The data is organized into separate directories for each table, making it easy to query and analyze later.

### Code Snippet Example

Here's a simplified example of how the ETL process is implemented in the `etl.py` script:

>```python
>from pyspark.sql import SparkSession
>from pyspark.sql.functions import col, to_timestamp
>
># Initialize Spark session
>spark = SparkSession.builder \
>    .appName("Sparkify ETL") \
>    .getOrCreate()
>
># Load song data
>song_data = spark.read.json("s3://udacity-dend/song_data/*/*/*.json")
>
># Load log data
>log_data = spark.read.json("s3://udacity-dend/log_data/*/*/*.json")
>
># Transform songplays table
>songplays_table = log_data \
>    .filter(col("page") == "NextSong") \
>    .join(song_data, log_data.song == song_data.title, "left") \
>    .select("userId", "level", "song_id", "artist_id", "sessionId", "location", "userAgent", "start_time")
>
># Write songplays table to S3
>songplays_table.write.partitionBy("year", "month").parquet("s3://your-bucket/songplays/")
>```

### Conclusion

This project demonstrates the effective use of Apache Spark to build a scalable data lake for Sparkify. By transitioning to a data lake architecture, Sparkify can efficiently analyze user behavior and song popularity, enabling better decision-making and insights. The ETL pipeline ensures that data is consistently processed and made available for analytics, supporting the growth of the music streaming service.

The implementation showcases the power of Spark for big data processing, allowing for real-time analytics and insights. The use of Parquet format enhances performance and storage efficiency, making it a suitable choice for data lakes.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Data-Lake-with-Spark).

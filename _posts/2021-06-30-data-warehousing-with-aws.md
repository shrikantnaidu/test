---
layout: post
title: Data Warehousing with AWS
date: 2021-06-30T 14:30:00 +03:00
description: "Building an ETL pipeline to extract, transform, and load data into AWS Redshift for a music streaming startup, enabling efficient analysis of user activity data."
image: "assets/images/masonary-post/aws.png"
categories: 
  - "Data Engineering"
---
## [Data Warehousing with AWS](https://github.com/shrikantnaidu/Data-Warehousing-with-AWS)
---

### Project Overview

Sparkify, a music streaming startup, has experienced significant growth in its user base and song database. To leverage this data for insights, they aim to transition their processes and data to the cloud. The existing data resides in Amazon S3, consisting of JSON logs on user activity and JSON metadata about the songs in their app.

The objective of this project is to build an ETL (Extract, Transform, Load) pipeline that extracts data from S3, stages it in AWS Redshift, and transforms it into a set of dimensional tables for the analytics team to derive insights into user listening patterns.

### Technical Architecture

#### Project Structure

>```
> data_warehousing_aws/
>│
>├── create_tables.py          # Database and table creation script
>├── etl.py                    # ETL pipeline script
>├── sql_queries.py            # SQL queries for CRUD operations
>├── create_cluster.ipynb      # Cloud resources setup and cleanup
>├── dwh.cfg                   # Credentials for cloud resources
>└── README.md                 # Project documentation
>```

### Core Components

• **create_tables.py**

- • Establishes a connection to the Redshift database.
- • Drops existing tables to ensure a fresh start.
- • Creates new tables using schema definitions.
- • Acts as a reset script for testing.

• **etl.py**

- • Implements the ETL pipeline.
- • Reads and processes files from `song_data` and `log_data`.
- • Transforms JSON data into appropriate formats.
- • Loads data into Redshift tables.

• **sql_queries.py**

- • Contains all SQL queries used throughout the project.
- • Includes CREATE, DROP, INSERT, and SELECT statements.
- • Centralizes query management for maintainability.

## Database Schema Design

The database uses a star schema optimized for song play analysis. This design prioritizes denormalization and simplifies queries while maintaining data integrity.

### Table Overview

| Table           ||| Description                   |
|-----------------|||-------------------------------|
| staging_events   ||| Staging table for events data |
| staging_songs    ||| Staging table for songs data  |
| songplays        ||| Table for the songs played    |
| users            ||| Table for the user data       |
| songs            ||| Table for the songs data      |
| artists          ||| Table for the artists data    |
| time             ||| Table for time-related data   |

### Fact Table

**songplays** - Records in event data associated with song plays.
>```sql
> CREATE TABLE IF NOT EXISTS songplays (
>     songplay_id SERIAL PRIMARY KEY,
>     start_time timestamp NOT NULL,
>     user_id int NOT NULL,
>     level varchar,
>     song_id varchar,
>     artist_id varchar,
>     session_id int,
>     location varchar,
>     user_agent varchar,
>     FOREIGN KEY (user_id) REFERENCES users (user_id),
>     FOREIGN KEY (song_id) REFERENCES songs (song_id),
>     FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
>     FOREIGN KEY (start_time) REFERENCES time (start_time)
> );
>```

### Dimension Tables

**users** - User information.
>```sql
>CREATE TABLE IF NOT EXISTS users (
>    user_id int PRIMARY KEY,
>    first_name varchar,
>    last_name varchar,
>    gender varchar,
>    level varchar
>);
>```

**songs** - Song metadata.
>```sql
>CREATE TABLE IF NOT EXISTS songs (
>    song_id varchar PRIMARY KEY,
>    title varchar,
>    artist_id varchar,
>    year int,
>    duration float
>);
>```

**artists** - Artist information.
>```sql
>CREATE TABLE IF NOT EXISTS artists (
>    artist_id varchar PRIMARY KEY,
>    name varchar,
>    location varchar,
>    latitude float,
>    longitude float
>);
>```

**time** - Timestamps of records broken down.
>```sql
>CREATE TABLE IF NOT EXISTS time (
>    start_time timestamp PRIMARY KEY,
>    hour int,
>    day int,
>    week int,
>   month int,
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

### 3. Loading Data into Redshift

The `COPY` command is a powerful feature of Amazon Redshift that allows for efficient loading of large datasets from Amazon S3 into Redshift tables. It is optimized for high throughput and can load data in parallel, making it suitable for big data applications.

#### Example of the COPY Command
>```sql
>COPY songplays
>FROM 's3://your-bucket/songplays/'
>IAM_ROLE 'arn:aws:iam::your-iam-role'
>FORMAT AS JSON 'auto';
>```

- a. **FROM**: Specifies the S3 bucket and path where the data files are located.
- b. **IAM_ROLE**: The IAM role that has permission to access the S3 bucket.
- c. **FORMAT AS JSON**: Indicates the format of the data being loaded. In this case, it is JSON.

### Redshift Capabilities

Amazon Redshift is a fully managed, petabyte-scale data warehouse service in the cloud. It allows you to run complex queries and perform analytics on large datasets quickly. Key capabilities include:

- a. **Scalability:** Easily scale your data warehouse up or down based on your needs.
- b. **Performance:** Redshift uses columnar storage and data compression to improve query performance.
- c. **Integration:** Seamlessly integrates with various AWS services, including S3, for data storage and retrieval.
- d. **Security:** Provides robust security features, including encryption and IAM roles for access control.

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
>SELECT EXTRACT(HOUR FROM start_time) AS hour, COUNT(*) as play_count
>FROM songplays
>GROUP BY hour
>ORDER BY play_count DESC;
>```

## How to Run

To set up and run the project, follow these steps:


#### 1. Set Up AWS Credentials:
>```
> - Create an IAM role with permissions to access S3 and Redshift.
> - Note the IAM role ARN for use in the `COPY` command.
>```

#### 2. Create and Activate a Python Virtual Environment:
>```bash
>python -m venv venv
>source venv/bin/activate  # On Windows: venv\Scripts\activate
>```

#### 3. Install Required Packages:
>```bash
>   pip install -r requirements.txt
>```

#### 4. Configure AWS Credentials:

Create a `dwh.cfg` file with your AWS credentials and Redshift cluster details. The file should look like this:
>```ini
>   [AWS]
>   KEY=your_aws_access_key
>   SECRET=your_aws_secret_key
>   [CLUSTER]
>   HOST=your_redshift_cluster_endpoint
>   DB_NAME=your_database_name
>   DB_USER=your_database_user
>   DB_PASSWORD=your_database_password
>   PORT=5439
>```

##### 5. Run create_tables.py to Set Up the Database and Tables:
>```bash
>python create_tables.py
>```

#### 6. Run the ETL Pipeline to Process and Load Data:
>```bash
>python etl.py
>```

## Key Achievements


- a. Designed an optimized star schema for efficient querying of music streaming data.
- b. Built a robust ETL pipeline that successfully processes and transforms JSON data.
- c. Implemented data validation and quality checks throughout the pipeline.
- d. Created a queryable database that enables complex analysis of user listening patterns.
- e. Achieved efficient data loading with minimal duplicate records.
- f. Implemented error handling and logging for pipeline monitoring.


## Technologies Used


• **Python 3.7+**

- • pandas for data manipulation
- • psycopg2 for PostgreSQL connection
- • json for parsing JSON files

• **AWS Redshift**

- • Columnar storage for efficient data retrieval
- • Scalability for handling large datasets

• **SQL**

- • DDL for schema definition
- • DML for data manipulation
- • Complex joins and aggregations


## Future Improvements

- a. Add data quality checks and constraints.
- b. Implement incremental loading.
- c. Add indexing for performance optimization.
- d. Create an automated testing suite.
- e. Implement logging and monitoring.
- f. Add a data visualization dashboard.

## Conclusion

This project successfully demonstrates the process of building a data warehousing solution on AWS for Sparkify. By leveraging AWS Redshift and an efficient ETL pipeline, the analytics team can now easily query and analyze user activity data, leading to valuable insights into user behavior and song popularity.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Data-Warehousing-with-AWS).

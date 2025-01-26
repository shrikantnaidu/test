---
layout: post
title: Data Pipelines with Airflow
date: 2021-07-17T 14:30:00 +03:00
description: "Automating ETL processes for Sparkify's data warehouse using Apache Airflow."
image: "assets/images/masonary-post/airflow.jpg"
categories: 
  - "Data Engineering"
---

## [Data Pipelines with Airflow](https://github.com/shrikantnaidu/Data-Pipelines-with-Airflow)

---

### Project Overview

Sparkify, a music streaming startup, has a growing need to manage and analyze vast amounts of user activity data. To facilitate this, the analytics team requires a robust ETL (Extract, Transform, Load) pipeline that can efficiently process and load data into a data lake. This project leverages Apache Airflow to orchestrate the ETL workflow, ensuring that data is consistently and reliably processed.

The challenge lies in automating the data pipeline to handle various data sources, transformations, and loading processes while maintaining data integrity and performance.

### Technical Architecture

The architecture of the data pipeline is designed to ensure data quality and efficient processing. The source datasets consist of JSON logs detailing user activity and song metadata. The project includes the following key components:

#### Project Structure

The project is organized into several key files:

<!-- - a. **dags/udac_example_dag.py**: Defines the DAG (Directed Acyclic Graph) for scheduling tasks and managing dependencies.
- b. **plugins/operators/**: Contains custom operator plugins used in the DAG.
- c. **stage_redshift.py**: Stages data from S3 to Redshift.
- d. **load_fact.py**: Loads data into the fact table in Redshift.
- e. **load_dimension.py**: Loads data into dimension tables in Redshift.
- f. **data_quality.py**: Performs data quality checks on Redshift tables.
- g. **plugins/helpers/**: Helper modules for the plugins.
- h. **README.md**: Documentation detailing the execution process and project setup. -->

>```
>data_pipelines_airflow/
>│
>├── dags/ # Directory for DAG definitions
>│  └── udac_example_dag.py # Defines the DAG for scheduling tasks
>│
>├── plugins/ # Custom plugins for Airflow
>│  ├── operators/ # Custom operator plugins
>│  └── helpers/ # Helper modules for the plugins
>│
>├── stage_redshift.py # Stages data from S3 to Redshift
>├── load_fact.py # Loads data into the fact table in Redshift
>├── load_dimension.py # Loads data into dimension tables in Redshift
>├── data_quality.py # Performs data quality checks on Redshift tables
>└── README.md # Project documentation
>```



### Core Components

- • **DAGs**: Directed Acyclic Graphs that define the workflow and task dependencies.
- • **Operators**: Custom operators that encapsulate the logic for specific tasks, such as loading data or performing transformations.
- • **Data Quality Checks**: Ensures the integrity of data after ETL steps, catching discrepancies early.


### Database Schema

#### Fact Table

**songplays** - Records of song plays, capturing user interactions.
>```sql
>CREATE TABLE IF NOT EXISTS songplays (
>songplay_id SERIAL PRIMARY KEY,
>start_time timestamp NOT NULL,
>user_id int NOT NULL,
>level varchar,
>song_id varchar,
>artist_id varchar,
>session_id int,
>location varchar,
>user_agent varchar,
>FOREIGN KEY (user_id) REFERENCES users (user_id),
>FOREIGN KEY (song_id) REFERENCES songs (song_id),
>FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
>FOREIGN KEY (start_time) REFERENCES time (start_time)
>);
>```

#### Dimension Tables

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

### ETL Pipeline Implementation

#### 1. Staging Data

The ETL pipeline first stages data from S3 to Redshift. The `stage_redshift.py` script handles this process, ensuring that data is loaded into staging tables before being transformed into the final schema.

#### 2. Loading Fact and Dimension Tables

The `load_fact.py` and `load_dimension.py` scripts are responsible for loading data into the fact and dimension tables, respectively. These scripts utilize SQL commands to insert data into the appropriate tables.

#### Example Queries and Results

##### 1. Most Active Users
>```sql
>sql
>SELECT u.first_name, u.last_name, COUNT() as play_count
>FROM songplays sp
>JOIN users u ON sp.user_id = u.user_id
>GROUP BY u.user_id, u.first_name, u.last_name
>ORDER BY play_count DESC
>LIMIT 5;
>```

##### 2. Popular Music Hours
>```sql
>SELECT t.hour, COUNT() as play_count
>FROM songplays sp
>JOIN time t ON sp.start_time = t.start_time
>GROUP BY t.hour
>ORDER BY play_count DESC;
>


### Key Achievements

- a. Designed a robust ETL pipeline using Apache Airflow for data orchestration.
- b. Implemented data quality checks to ensure data integrity.
- c. Created a scalable architecture that can handle various data sources and transformations.

### Technologies Used

- a. **Apache Airflow**: For orchestrating the ETL workflow.
- b. **PostgreSQL**: For storing the transformed data.
- c. **Python**: For scripting the ETL processes.

### Future Improvements

- a. Implement incremental loading to optimize data processing.
- b. Enhance monitoring and alerting for the ETL processes.
- c. Add more complex data transformations to support advanced analytics.

### Conclusion

This project exemplifies the power of Apache Airflow in automating and managing ETL processes for data warehouses. By implementing a robust data pipeline, Sparkify can efficiently process and analyze user activity data, leading to valuable insights into user behavior and song preferences. 

For more details, check out the complete implementation in the [GitHub repository](https://github.com/shrikantnaidu/Data-Pipelines-with-Airflow).



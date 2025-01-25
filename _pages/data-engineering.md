---
layout: page
title: Data Engineering Projects
permalink: /data-engineering/
image: 
---

Welcome to my data engineering projects portfolio. Here you'll find a collection of projects focused on building scalable data pipelines, designing efficient data models, and leveraging technologies like Apache Spark, PostgreSQL, and Apache Cassandra to process and analyze large datasets. Each project showcases my expertise in data engineering and my ability to tackle complex data challenges.


{% assign data_engineering_posts = site.posts | where_exp: "post", "post.categories contains 'Data Engineering'" %}

{% if data_engineering_posts.size > 0 %}
  {% for post in data_engineering_posts %}
#### [{{ post.title }}]({{ post.url | prepend: site.baseurl }})

{% if post.description %}
{{ post.description }}
{% endif %}

*Published on {{ post.date | date: '%B %d, %Y' }}*

{% if post.tags.size > 0 %}
**Technologies:** {% for tag in post.tags %}`#{{ tag }}` {% endfor %}
{% endif %}

  {% endfor %}
{% else %}
*No Data Engineering projects posted yet. Check back soon!*
{% endif %}
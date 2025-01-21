---
layout: page
title: Deep Learning Projects
permalink: /deep-learning/
---

Welcome to my deep learning projects portfolio. Here you'll find various projects involving neural networks, computer vision, natural language processing, and more. Each project demonstrates different aspects of deep learning applications and methodologies.


{% assign deep_learning_posts = site.posts | where_exp: "post", "post.categories contains 'Deep Learning'" %}

{% if deep_learning_posts.size > 0 %}
  {% for post in deep_learning_posts %}
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
*No Deep Learning projects posted yet. Check back soon!*
{% endif %}


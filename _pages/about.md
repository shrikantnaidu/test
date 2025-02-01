---
layout: about
title: About Me
permalink: /about/
# image: /assets/images/trp-sharpened.png
---

<h3 class="font-weight-light">Hello, I'm <span class="font-weight-bold">{{site.author.name}}</span></h3>

Welcome to my digital space! I'm a seasoned Data Scientist with a passion for crafting innovative, data-driven solutions to tackle complex business challenges. Currently, I lead a high-performance team responsible for architecting and deploying scalable data solutions that address complex business challenges. 

With extensive experience in data engineering, advanced analytics, machine learning, and deep learning, I specialize in transforming raw data into actionable insights that drive strategic decision-making and operational efficiency.

My professional journey began at Leadzpipe, where I developed end-to-end data pipelines to extract, model, and analyze Google AdWords campaign data, optimizing client performance. This work was recognized with the prestigious Aegis Graham Bell Award in 2020, underscoring my commitment to excellence in data-driven problem-solving.

With a solid foundation in Computer Science and a deep technical proficiency in data science, I am driven by a passion for leveraging cutting-edge technologies to solve intricate problems. My expertise spans data architecture, pipeline optimization, machine learning, and geospatial analytics, with a focus on delivering solutions that are not only innovative but also highly performant.

Have an unsolvable problem? I'm just an mail away – **shrikantnaidu777@gmail.com**.

---
<h3>Professional Experience</h3>

<div style="max-width: 800px; margin: auto; position: relative;">
    <div style="border-left: 2px solid black; padding-left: 30px; position: relative;">
        <div style="margin-bottom: 40px; position: relative;">
            <div style="position: absolute; left: -40px; top: 0; width: 20px; height: 20px; background-color: black; border-radius: 50%; border: 2px solid white; z-index: 1;"></div>
            <h5 style="margin-top: 0;">Manager - Data Science & Engineering at Loylty Rewardz</h5>
            <p><em>March 2024 - Present</em></p>
            <!-- <p>Leading the fulfillment of data science and engineering requirements crucial to the business.</p> -->
            <hr style="border: 1px solid #ccc; margin-top: 10px;">
        </div>
        <div style="margin-bottom: 40px; position: relative;">
            <div style="position: absolute; left: -40px; top: 0; width: 20px; height: 20px; background-color: black; border-radius: 50%; border: 2px solid white; z-index: 1;"></div>
            <h5 style="margin-top: 0;">Data Scientist II at Loylty Rewardz</h5>
            <p><em>March 2023 - March 2024</em></p>
            <!-- <p>Leading the fulfillment of data science and engineering requirements crucial to the business.</p> -->
            <hr style="border: 1px solid #ccc; margin-top: 10px;">
        </div>
        <div style="margin-bottom: 40px; position: relative;">
            <div style="position: absolute; left: -40px; top: 0; width: 20px; height: 20px; background-color: black; border-radius: 50%; border: 2px solid white; z-index: 1;"></div>
            <h5 style="margin-top: 0;">Data Scientist at Predoole Analytics</h5>
            <p><em>October 2022 - March 2023</em></p>
            <!-- <p>Revamped and optimized data pipelines in the insurance sector.</p> -->
            <hr style="border: 1px solid #ccc; margin-top: 10px;">
        </div>
        <div style="margin-bottom: 40px; position: relative;">
            <div style="position: absolute; left: -40px; top: 0; width: 20px; height: 20px; background-color: black; border-radius: 50%; border: 2px solid white; z-index: 1;"></div>
            <h5 style="margin-top: 0;">Data Analyst at Medly</h5>
            <p><em>October 2020 - September 2022</em></p>
            <!-- <p>Empowered stakeholders with data-driven decision-making by leveraging analytics and machine learning techniques.</p> -->
            <hr style="border: 1px solid #ccc; margin-top: 10px;">
        </div>
        <div style="margin-bottom: 40px; position: relative;">
            <div style="position: absolute; left: -40px; top: 0; width: 20px; height: 20px; background-color: black; border-radius: 50%; border: 2px solid white; z-index: 1;"></div>
            <h5 style="margin-top: 0;">Data Science Intern at Leadzpipe</h5>
            <p><em>October 2019 - May 2020</em></p>
            <!-- <p>Extracted campaign data from Google AdWords and modeled it for NoSQL databases.</p> -->
            <hr style="border: 1px solid #ccc; margin-top: 10px;">
        </div>
    </div>
</div>
<hr>

### Awards 

<div class="image-slider">
    <div class="slider">
        <div class="slide">
            <img src="/assets/images/2-sharpened.png" alt="Image 1">
            <div class="caption">Aegis Graham Bell Awards 2020</div>
        </div>
        <div class="slide">
            <img src="/assets/images/img-2.png" alt="Image 2">
            <div class="caption">Extraordinary Diligence Award 2024</div>
        </div>
        <!-- <div class="slide">
            <img src="/assets/images/image3.jpg" alt="Image 3">
            <div class="caption">Caption for Image 3</div>
        </div> -->
    </div>
    <button class="prev" onclick="moveSlide(-1)">&#10094;</button>
    <button class="next" onclick="moveSlide(1)">&#10095;</button>
</div>

<style>
.image-slider {
    position: relative;
    max-width: 100%;
    margin: auto;
}

.slider {
    display: flex;
    overflow: hidden;
}

.slide {
    min-width: 100%;
    transition: transform 0.5s ease;
    text-align: center;
}

.slide img {
    max-width: 100%;
    max-height: 400px;
    height: auto;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}

.caption {
    margin-top: 10px;
    font-size: 14px;
    color: #555;
}

button {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background-color: rgba(255, 255, 255, 0.8);
    border: none;
    cursor: pointer;
}

.prev {
    left: 10px;
}

.next {
    right: 10px;
}
</style>

<script>
let currentSlide = 0;

function showSlide(index) {
    const slides = document.querySelectorAll('.slide');
    if (index >= slides.length) {
        currentSlide = 0;
    } else if (index < 0) {
        currentSlide = slides.length - 1;
    } else {
        currentSlide = index;
    }
    const offset = -currentSlide * 100;
    slides.forEach(slide => {
        slide.style.transform = `translateX(${offset}%)`;
    });
}

function moveSlide(direction) {
    showSlide(currentSlide + direction);
}

// Initialize the slider
showSlide(currentSlide);
</script>

<hr>

<h3>Socials</h3>

<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 15px; margin: 20px 0;">
    <a href="https://x.com/shrikantnaiidu" target="_blank" rel="noreferrer">
        <img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/socials/twitter.svg" width="32" height="32" alt="X" />
    </a>
    <a href="https://www.linkedin.com/in/shrikant-naidu/" target="_blank" rel="noreferrer">
        <img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/socials/linkedin.svg" width="32" height="32" alt="LinkedIn" />
    </a>
    <a href="https://app.datacamp.com/profile/shrikantnaidu777" target="_blank" rel="noreferrer">
        <img src="https://www.svgrepo.com/show/349332/datacamp.svg" width="32" height="32" alt="DataCamp" />
    </a>
    <a href="https://steamcommunity.com/id/shrikantnaidu/" target="_blank" rel="noreferrer">
        <img src="https://www.vectorlogo.zone/logos/steampowered/steampowered-icon.svg" width="32" height="32" alt="Steam" />
    </a>
    <a href="https://wandb.ai/skn97" target="_blank" rel="noreferrer">
        <img src="https://www.vectorlogo.zone/logos/wandbai/wandbai-official.svg" width="100" height="35" alt="Weights & Biases" />
    </a>
    <a href="https://huggingface.co/shrikantnaidu" target="_blank" rel="noreferrer">
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="32" height="32" alt="Hugging Face" />
    </a>
</div>

<hr>
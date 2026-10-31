---
title: Student Face Registration Cloud Portal
emoji: 🎓
colorFrom: yellow
colorTo: amber
sdk: docker
app_port: 7860
pinned: false
---

# سامانه ثبت‌نام ابری دانشجویان | Student Registration Cloud Portal

این سامانه به صورت کانتینری و مستقل روی سرویس رایگان **Hugging Face Spaces** با معماری داکر اجرا می‌شود.

## ویژگی‌ها:
- رابط کاربری ریسپانسیو و فارسی (RTL) مخصوص موبایل و دسکتاپ در مسیر `/` و `/register`
- اعتبارسنجی بیومتریک چهره با هوش مصنوعی (`face_recognition` و `dlib`)
- بررسی تصاویر تکی، عدم وجود چهره یا وجود چند چهره در تصویر
- جلوگیری از ثبت چهره یا کد دانشجویی تکراری
- خروجی همگام‌سازی ابری با یک کلیک از طریق اندپوینت ایمن:
  `GET /api/cloud/export?token=mahyar123`

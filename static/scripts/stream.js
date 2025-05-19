const video = document.getElementById('video');
video.onload = () => console.log('Video stream loaded');
video.onerror = () => {
    console.error('Error loading video stream. Retrying...');
    setTimeout(() => { video.src = '/video_feed'; }, 2000);
};
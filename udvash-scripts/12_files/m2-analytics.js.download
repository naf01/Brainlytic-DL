function base64UrlEncode(value) {
    let bytes = new TextEncoder().encode(value);
    let base64String = btoa(String.fromCharCode.apply(null, bytes));
    return base64String.replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

document.addEventListener("DOMContentLoaded", function () {
    let currentUrl = window.location.href;
    currentUrl = base64UrlEncode(currentUrl);
    let anlyticsUrl = `https://gateway.onnorokomprojukti.com/AnalyticsApi/Analytics/Log/${currentUrl}`;

    document.getElementById('AnalyticsIFrame').setAttribute('src', anlyticsUrl);
});
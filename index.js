const path = require('path');
const express = require('express');
const app = express();

const PORT = process.env.PORT || 5000;

// Static Assests
app.use(express.static(path.join(__dirname, 'public')));

// Form Submission
// Handle Form Data
app.use(express.urlencoded({ extended: true }));

// Get Request
const api_route = require('./router/route');

app.use('/', api_route);

app.listen(PORT, () => {
    console.log(`Application listening on port ${PORT}!`);
});
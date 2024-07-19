// Class representing a registration for an event
class Registration {
    constructor(eventName, eventDate, organizer, capacity, email) {
        this.eventName = eventName; // Event name
        this.eventDate = eventDate; // Event date
        this.organizer = organizer; // Organizer's name
        this.capacity = capacity; // Capacity of the event
        this.cost = capacity * 15; // Cost calculation based on capacity
        this.email = email; // Organizer's email
    }
}

// Class managing multiple event registrations
class EventManager {
    constructor() {
        this.registrations = this.loadRegistrations(); // Load existing registrations from localStorage
        this.updateStatistics(); // Update statistics on initialization
        this.renderTable(); // Render the registrations table on initialization
    }

    // Method to add a new registration
    addRegistration(registration) {
        this.registrations.push(registration); // Add new registration to the list
        this.saveRegistrations(); // Save updated registrations to localStorage
        this.renderTable(); // Re-render the registrations table
        this.updateStatistics(); // Update statistics after adding registration
    }

    // Method to delete a registration by index
    deleteRegistration(index) {
        this.registrations.splice(index, 1); // Remove registration from the list
        this.saveRegistrations(); // Save updated registrations to localStorage
        this.renderTable(); // Re-render the registrations table
        this.updateStatistics(); // Update statistics after deleting registration
    }

    // Method to save registrations to localStorage
    saveRegistrations() {
        localStorage.setItem('registrations', JSON.stringify(this.registrations)); // Store registrations as JSON string
    }

    // Method to load registrations from localStorage
    loadRegistrations() {
        const data = localStorage.getItem('registrations'); // Retrieve registrations data from localStorage
        return data ? JSON.parse(data) : []; // Parse JSON data or return an empty array if no data exists
    }

    // Method to render the registrations table in the UI
    renderTable() {
        const tableBody = document.querySelector('#eventTable tbody'); // Select the table body element

        tableBody.innerHTML = ''; // Clear existing table rows

        // Loop through registrations and create table rows
        this.registrations.forEach((registration, index) => {
            const row = document.createElement('tr'); // Create a new table row

            // Loop through registration properties and create table cells
            Object.values(registration).forEach(value => {
                const cell = document.createElement('td'); // Create a new table cell
                cell.textContent = value; // Set cell content to registration property value
                row.appendChild(cell); // Append cell to the row
            });

            // Create a cell for actions (delete button)
            const actionsCell = document.createElement('td');
            const deleteButton = document.createElement('button'); // Create a delete button
            deleteButton.textContent = 'Delete'; // Set button text
            deleteButton.classList.add('btn', 'btn-delete'); // Add classes to button
            deleteButton.onclick = () => this.deleteRegistration(index); // Attach delete action on click

            actionsCell.appendChild(deleteButton); // Append delete button to actions cell
            row.appendChild(actionsCell); // Append actions cell to the row

            tableBody.appendChild(row); // Append row to the table body
        });
    }

    // Method to update statistics (total events and average cost)
    updateStatistics() {
        const totalEvents = this.registrations.length; // Total number of events
        const totalCost = this.registrations.reduce((acc, reg) => acc + reg.cost, 0); // Total cost of all events
        const averageCost = totalEvents > 0 ? (totalCost / totalEvents).toFixed(2) : 0; // Average cost per event

        // Update total events and average cost in the UI
        document.getElementById('totalEvents').textContent = `Total Events: ${totalEvents}`;
        document.getElementById('averageCost').textContent = `Average Cost: ${averageCost} Euros`;
    }
}

const eventManager = new EventManager(); // Initialize the EventManager instance

// Function to handle adding a new registration from the form
function addRegistration() {
    // Get form input values
    const eventName = document.getElementById('eventName').value;
    const eventDate = document.getElementById('eventDate').value;
    const organizer = document.getElementById('organizer').value;
    const capacity = parseInt(document.getElementById('capacity').value);
    const email = document.getElementById('email').value;

    const errorMessage = document.getElementById('errorMessage'); // Error message element
    errorMessage.textContent = ''; // Clear previous error message

    // Validation checks for form inputs
    if (!eventName) {
        errorMessage.textContent = 'Please enter an Event Name.';
        return;
    }

    if (!eventDate) {
        errorMessage.textContent = 'Please select a Date for the event.';
        return;
    }

    const selectedDate = new Date(eventDate); // Convert event date string to Date object
    const currentDate = new Date(); // Current date object
    currentDate.setHours(0, 0, 0, 0); // Set hours to 0 for both dates

    if (selectedDate <= currentDate) {
        errorMessage.textContent = 'Please enter a future Date for the event.';
        return;
    }

    if (!organizer) {
        errorMessage.textContent = 'Please enter the Organizer\'s Name.';
        return;
    }

    if (!capacity || isNaN(capacity) || capacity <= 0) {
        errorMessage.textContent = 'Please enter a valid Capacity.';
        return;
    }

    if (!email || !validateEmail(email)) {
        errorMessage.textContent = 'Please enter a valid email address.';
        return;
    }

    // Create new Registration object with form values
    const registration = new Registration(eventName, eventDate, organizer, capacity, email);
    eventManager.addRegistration(registration); // Add registration using EventManager
    document.getElementById('registrationForm').reset(); // Reset form inputs after submission
}

// Function to validate email format using regular expression
function validateEmail(email) {
    const re = /\S+@\S+\.\S+/; // Regular expression for email format
    return re.test(email); // Return true if email matches regex pattern
}

// Function to print the registrations table
function printTable() {
    window.print(); // Print current page
}

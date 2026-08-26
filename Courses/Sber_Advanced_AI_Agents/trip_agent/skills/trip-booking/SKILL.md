---
name: trip-booking
description:
  Help a traveler who has already picked a destination move toward booking by
  gathering trip details, then using search_flights/search_hotels to find
  options and book_trip to finalize — all backed by mock providers, no real
  inventory or payment.
version: 1.1.0
---

# Purpose
Help a traveler who has already picked a destination move toward booking a
flight and hotel, using the search_flights, search_hotels, and book_trip
tools rather than inventing details yourself.

# Instructions
- Confirm the destination and ask for whatever is still missing: travel
  dates, trip length, number of travelers, and rough budget.
- Once you have enough to work with, call search_flights and search_hotels
  with those details. Each delegates to its own specialized search agent —
  it may come back with concrete options, or a clarifying question if
  something critical is still missing; relay that question to the user.
- Present the returned options plainly. Don't invent additional flights,
  hotels, or prices beyond what the tools returned.
- Once the user picks a flight and hotel, call book_trip with the
  destination, dates, and the chosen flight/hotel to finalize, then confirm
  the booking (including the confirmation code it returns) back to the user.
- Keep responses to a few sentences — this is a conversation, not an
  itinerary document.

# Constraints
- Don't invent flight/hotel options or prices yourself — only report what
  search_flights/search_hotels/book_trip actually return.
- Don't call book_trip until the user has confirmed a specific flight and
  hotel.
- Don't re-litigate the destination choice — that already happened in a
  prior step.

# Expected output
Plain conversational reply. No structured output.

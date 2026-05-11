# Reminder pinging is handled by the reminder_loop task in main.py.
# That loop runs every 60 seconds, checks memory/ryo.db for due reminders,
# and DMs the user directly via discord.ext.tasks.

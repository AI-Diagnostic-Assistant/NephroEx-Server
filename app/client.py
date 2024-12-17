from supabase import Client, create_client

def create_sb_client():
    url: str = 'https://lwybipvgkqqcuevmabdl.supabase.co'
    key: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3eWJpcHZna3FxY3Vldm1hYmRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mjk4NjE2NjMsImV4cCI6MjA0NTQzNzY2M30.aajawhgjIHCije-6BICug6qVuhcuK740rHXmSuQkKO0'
    return create_client(url, key)
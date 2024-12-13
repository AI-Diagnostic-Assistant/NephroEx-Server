from supabase import Client, create_client

url: str = 'https://lwybipvgkqqcuevmabdl.supabase.co'
key: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3eWJpcHZna3FxY3Vldm1hYmRsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyOTg2MTY2MywiZXhwIjoyMDQ1NDM3NjYzfQ.pMjAsxj8xM57VlFcJQ4zXR_XIoXkhJ-f64fkwsjZ9xQ'
supabase_client: Client = create_client(url, key)
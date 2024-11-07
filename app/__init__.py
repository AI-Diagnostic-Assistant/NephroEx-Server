from flask import Flask
from flask_supabase import Supabase

app = Flask(__name__)

app.config['SUPABASE_URL'] = 'https://lwybipvgkqqcuevmabdl.supabase.co'
app.config['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3eWJpcHZna3FxY3Vldm1hYmRsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyOTg2MTY2MywiZXhwIjoyMDQ1NDM3NjYzfQ.pMjAsxj8xM57VlFcJQ4zXR_XIoXkhJ-f64fkwsjZ9xQ'

supabase_extension = Supabase(app)

from app import routes


if __name__ == '__main__':
    app.run(debug=True)

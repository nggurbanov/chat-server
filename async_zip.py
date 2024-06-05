import asyncio
import aiofiles
import os
import zipfile

async def zip(folder_path, zip_path):
    loop = asyncio.get_event_loop()

    def zip_task(file_data):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, data in file_data.items():
                arcname = os.path.relpath(file_path, folder_path)
                zipf.writestr(arcname, data)
    
    file_data = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                file_data[file_path] = data
    
    await loop.run_in_executor(None, zip_task, file_data)

async def unzip(zip_path, extract_to):
    loop = asyncio.get_event_loop()

    def unzip_task():
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)
            return zipf.namelist()

    file_list = await loop.run_in_executor(None, unzip_task)

    async def extract_file(file_name):
        source_path = os.path.join(extract_to, file_name)
        async with aiofiles.open(source_path, 'rb') as f:
            data = await f.read()
        
        async with aiofiles.open(source_path, 'wb') as f:
            await f.write(data)

    await asyncio.gather(*(extract_file(file) for file in file_list))

import mongoose from "mongoose";

type ConnectionObject={
    isConnected?:number
}

const connection : ConnectionObject = {}

async function dbConnect(): Promise<void> {
    if(connection.isConnected){
        console.log("Already Connected");
        return
    }
    try {
        const db = await mongoose.connect(process.env.MONGO_DB_URI! || '')
        // console.log(db.connections);
        connection.isConnected=db.connections[0].readyState
        console.log("DB succesfully connected");

    } catch (error) {
        console.log("DB connection failed",error);
        process.exit()
    }
}


export default dbConnect;
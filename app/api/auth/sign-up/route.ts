import {NextRequest, NextResponse} from "next/server";
import dbConnect from "@/lib/db";

export async function POST(req: NextRequest) {
    try {
        await dbConnect();

        const { firstName, lastName, email, password } = await req.json();

    } catch (e) {
        return NextResponse.json(
            { error: e },
            { status: 500 }
        )
    }
}
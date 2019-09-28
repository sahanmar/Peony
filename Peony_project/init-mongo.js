db.createUser(
    {
        user: "User",
        pwd: "Pass",
        roles: [
            {
                role: "readWrite",
                db: "Peony-MongoDb"
            }
        ]

    }
)
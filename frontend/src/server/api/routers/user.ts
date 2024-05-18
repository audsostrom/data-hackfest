import {createTRPCRouter, protectedProcedure} from "~/server/api/trpc";
import {z} from "zod";
import {eq} from "drizzle-orm";
import {users} from "~/server/db/schema";

export const userRouter = createTRPCRouter({
    // defines purpose, protected = requires login
    byId: protectedProcedure
        // validates input
        .input(z.number())
        // sets up query with context and input
        .query(({ctx, input}) => {
            // drizzle orm select
            // https://orm.drizzle.team/docs/rqb
            return ctx.db.query.users.findFirst({
                where: eq(users.id, input),
            })
        }),

    getAll: protectedProcedure.query(({ctx}) => {
        return ctx.db.query.users.findMany();
    }),
});